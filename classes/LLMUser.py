import json
import re
import time
from classes.PromptBuilder import PromptBuilder


class LLMUser:
    """
    High-level interface for:
    - verbalizing choices into facts
    - extracting arguments from text
    - detecting pairwise argument relations
    """

    def __init__(self, llm):
        self.llm = llm
        self.debug = True

    # ---------------------------
    # Wikipedia retrieval
    # ---------------------------

    def extract_unified_arguments(self, text, hypotheses, previous_arguments, max_new=5):

        prompt = PromptBuilder.unified_argument_prompt(text, hypotheses, previous_arguments, max_new)
        response = self.llm.send_prompt(prompt)

        raw = getattr(response, "raw_text", "").strip()

        # --- Remove code fences ---
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw).strip()

        # --- Find first JSON start ---
        m = re.search(r"(\[|\{)", raw)
        if not m:
            if self.debug:
                print("⚠ unified extractor: no JSON start found, returning empty.")
            return []

        raw = raw[m.start():]

        # --- Fix common formatting issues before JSON parse ---
        raw = raw.replace("\r", "").replace("\\", "\\\\")
        if raw.count("[") > raw.count("]"):
            raw += "]"
        if raw.count("{") > raw.count("}"):
            raw += "}"

        # --- Try primary JSON parse ---
        try:
            data = json.loads(raw)
        except Exception as e:
            if self.debug:
                print(f"⚠ unified extractor parse failed: {e}. Attempting repair using Ollama…")
                print(f"Raw snippet: {raw[:300]}")

            # --- JSON REPAIR via OLLAMA -----------------------------------------
            repair_prompt = f"""
    You fix broken JSON. Output ONLY valid JSON, no explanations.

    Target JSON specification:
    A dictionary with:
      arguments: [string]
      relations: [string or null]
      targets:   [string or null]

    or a list of such dictionaries.

    Here is the broken JSON:
    {raw}
    """

            try:
                import ollama
                repair_result = ollama.generate(model="qwen3:14b", prompt=repair_prompt)
                repaired_raw = repair_result["response"]
            except Exception as e2:
                if self.debug:
                    print(f"❌ Ollama repair failed: {e2}")
                return []

            # --- Try parsing repaired JSON ---
            try:
                data = json.loads(repaired_raw)
                if self.debug:
                    print("🔧 JSON successfully repaired by Ollama.")
            except Exception as e3:
                if self.debug:
                    print(f"⚠ JSON repair parse still failed: {e3}")
                    print(f"Repaired snippet: {repaired_raw[:300]}")
                return []

        # ======================================================
        #  ITEM-LEVEL NORMALIZATION & REPAIR
        # ======================================================

        def normalize_item(item):
            """
            Repairs malformed argument objects.
            Maps alternate key names → canonical keys.
            Ensures missing fields are filled.
            """

            text_val = (
                item.get("text")
                or item.get("argument")
                or item.get("arg")
                or item.get("claim")
                or item.get("content")
                or ""
            )

            relation_val = (
                item.get("relation")
                or item.get("stance")
                or item.get("type")
                or "indifferent"
            )

            target_val = (
                item.get("target")
                or item.get("about")
                or item.get("subject")
                or item.get("object")
                or ""
            )

            return {
                "text": text_val,
                "relation": relation_val,
                "target": target_val,
            }

        # ======================================================
        # Handling dictionary structure (your original schema)
        # ======================================================

        if isinstance(data, dict):
            for key in ["arguments", "relations", "targets"]:
                if key not in data or not isinstance(data[key], list):
                    data[key] = []

            # pad missing fields
            n_args = len(data["arguments"])
            data["relations"] = data["relations"] + [None] * (n_args - len(data["relations"]))
            data["targets"]   = data["targets"]   + [None] * (n_args - len(data["targets"]))

            out = []
            for txt, rel, tgt in zip(data["arguments"], data["relations"], data["targets"]):
                out.append(normalize_item({
                    "text": txt,
                    "relation": rel,
                    "target": tgt
                }))
            return out

        # ======================================================
        # Handling list-type outputs
        # ======================================================
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, dict):
                    out.append(normalize_item(item))
            return out

        return []





    def get_candidate_pages(self, question, choices, max_pages=5):
        """
        Generate a list of relevant Wikipedia page titles using the LLM.

        Improvements:
        - If the primary LLM returns broken JSON, call a dedicated JSON-repair model.
        - Defensive logging, fallbacks, and safe return values preserved.
        """

        prompt = PromptBuilder.wikipedia_retrieval_prompt(question, choices, max_pages)

        try:
            # --- Primary LLM call ---
            t0 = time.time()
            response = self.llm.send_prompt(prompt)
            t1 = time.time()

            raw = getattr(response, "raw_text", "") if response else ""
            print(f"⏱ get_candidate_pages took {t1 - t0:.2f}s — response length {len(raw)}")

            # --- Try initial JSON parse ---
            try:
                pages = json.loads(raw)
                if isinstance(pages, list):
                    return pages[:max_pages]
            except Exception as e:
                print(f"⚠️ JSON parse failed: {e}. Attempting repair...", flush=True)

            # --- JSON Repair Step ------------------------------------------------
            if raw.strip():
                repair_prompt = f"""
    You are a JSON repair model. Your job is to take *invalid or broken JSON* 
    and output *only valid JSON*, following this schema:

    Schema: A JSON list of strings representing Wikipedia page titles.
    Example: ["Cat", "Dog", "Fox"]

    Broken JSON:
    {raw}

    Return ONLY repaired JSON, with no explanations.
    """

                try:
                    repair_response = self.json_repair_model.send_prompt(repair_prompt)
                    repaired_raw = getattr(repair_response, "raw_text", "")

                    try:
                        repaired_pages = json.loads(repaired_raw)
                        if isinstance(repaired_pages, list):
                            print("🔧 JSON successfully repaired.", flush=True)
                            return repaired_pages[:max_pages]
                    except Exception as e:
                        print(f"⚠️ Repaired JSON parse still failed: {e}. Snippet: {repaired_raw[:200]!r}")
                except Exception as e:
                    print(f"❌ JSON-repair model failed: {e}")

            # --- Fallback: newline-separated titles ---
            if raw and "\n" in raw:
                lines = [l.strip() for l in raw.splitlines() if l.strip()]
                return lines[:max_pages]

            # --- Last-resort fallback ---
            return [raw] if raw else []

        except Exception as e:
            print(f"❌ Exception in get_candidate_pages: {e}")
            traceback.print_exc()
            return []


    # ---------------------------
    # Argument extraction
    # ---------------------------
    def extract_arguments_with_ollama(self, text: str, hypotheses=None, max_arguments=3) -> list:
        if hypotheses is None:
            hypotheses = []

        prompt = PromptBuilder.argument_extraction_prompt(text, hypotheses, max_arguments)
        response = self.llm.send_prompt(prompt)

        raw = response.raw_text.strip()
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw).strip()  # remove markdown fences

        try:
            args = json.loads(raw)
            cleaned_args = []

            if isinstance(args, list):
                for arg in args:
                    if isinstance(arg, list):
                        # ✅ treat each inner item as a separate argument
                        for sub in arg:
                            sub_clean = str(sub).strip()
                            if sub_clean and sub_clean not in cleaned_args:
                                cleaned_args.append(sub_clean)
                    else:
                        # Single string argument
                        arg_clean = str(arg).strip()
                        if arg_clean and arg_clean not in cleaned_args:
                            cleaned_args.append(arg_clean)

                # Limit total number of extracted arguments
                return cleaned_args[:max_arguments]

        except Exception:
            # fallback: extract sentences if JSON parsing fails
            sentences = [
                s.strip() for s in re.split(r'(?<=[.!?])\s+', raw)
                if len(s.strip()) > 4
            ]
            return sentences[:max_arguments]

        return []




    # ---------------------------
    # Relation detection
    # ---------------------------
    def detect_argument_relations_pairwise(self, arguments: list) -> dict:
        """
        Compare all pairs of arguments and detect relations.
        Returns a dict like {"0-1": "support", "1-2": "attack", ...}.
        """
        relations = {}
        for i, arg_a in enumerate(arguments):
            for j, arg_b in enumerate(arguments):
                if i == j:
                    continue

                prompt = PromptBuilder.pairwise_relation_prompt(arg_a, arg_b)
                raw_response = self.llm.send_prompt(prompt).raw_text

                try:
                    result = json.loads(raw_response)
                    rel = result.get("relation", "indifferent")
                    relations[f"{i}-{j}"] = rel
                except Exception:
                    relations[f"{i}-{j}"] = "indifferent"

        return relations

    # ---------------------------
    # Verbalize choices
    # ---------------------------
    def verbalize_choice_dataset(self, question: str, choice: str) -> str:
        """
        Transform a multiple-choice question and option into a concise factual statement.
        """
        prompt = f"""
Transform the following multiple-choice question and option into a concise factual statement, making the option the subject of the sentence. 
Include all relevant details from the question, such as numbers, actions, conditions, and context. 
Return only the factual statement as plain text.

Example 1:
Question: "Compounds that are capable of accepting electrons, such as O2 or F2, are called what?"
Option: "oxidants"
Output: "Oxidants are compounds capable of accepting electrons, such as O2 or F2."

Example 2:
Question: "A physicist wants to determine the speed a car must reach to jump over a ramp. 
The physicist conducts three trials. In trials two and three, the speed of the car is increased by 20 miles per hour. 
What is the physicist investigating when he changes the speed?"
Option: "independent (manipulated) variable"
Output: "The independent (manipulated) variable is the speed of the car that the physicist changes in each trial to determine the speed the car must reach to jump over the ramp, increasing it by 20 miles per hour in trials two and three."

Example 3:
Question: "A 40-year-old man presents with 5 days of productive cough and fever. 
Pseudomonas aeruginosa is isolated from his sputum culture. 
Which of the following is the most appropriate initial antibiotic therapy?"
Option: "Piperacillin-tazobactam"
Output: "Piperacillin-tazobactam is the most appropriate initial antibiotic therapy for a 40-year-old man with 5 days of productive cough and fever in whom Pseudomonas aeruginosa is isolated from the sputum culture."

Now transform this one:
Question: "{question}"
Option: "{choice}"
"""
        response = self.llm.send_prompt(prompt)
        return response.raw_text.strip().strip("`").strip()









