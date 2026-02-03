import json

class PromptBuilder:
    """
    Builds and manages prompts for different LLM tasks.
    Each method returns a formatted text prompt that can be
    sent to an LLM (e.g., OllamaChat).
    """

    # ------------------------------------------------------------
    #  WIKIPEDIA RETRIEVAL PROMPT
    # ------------------------------------------------------------
    @staticmethod
    def wikipedia_retrieval_prompt(question: str, choices: list, max_pages: int = 5) -> str:
        return f'''
        You are a high-precision Wikipedia retrieval agent.

        Your task:
        1. Understand the question and multiple-choice options.
        2. Identify the key concepts, entities, or facts needed to answer the question.
        3. Return the most relevant Wikipedia pages that could contain the answer.
        4. Score each page with a relevance score between 0 and 1.
           - 1.0 = extremely relevant / almost certainly needed
           - 0.5 = possibly relevant
           - 0.0 = not relevant (do NOT output)

        Strict rules:
        - Suggest at most {max_pages} pages.
        - If a page title is ambiguous, choose the most relevant disambiguated form
          (e.g., "Mercury (planet)" instead of "Mercury").
        - Prefer specific factual pages over overly broad topics.
        - Do NOT fabricate nonexistent Wikipedia titles.
        - Output must be valid JSON. No explanations. No extra text.

        Input:
        Question:
        {question}

        Choices:
        {json.dumps(choices, ensure_ascii=False, indent=2)}

        Output format (JSON list of [title, relevance]):
        [
          ["Page Title A", 0.92],
          ["Page Title B", 0.76]
        ]

        Respond ONLY with the JSON list.
        ''' 


    def unified_argument_prompt(text: str, hypotheses: list, prev_arguments: list, max_new: int = 5) -> str:
        return f"""
            You are an expert argument-mining system.

            Your task is to extract **at most {max_new} new arguments** from the text and identify their **direct logical relation** to either:
            1. a hypothesis, or
            2. a previously extracted argument.

            ======================
             STRICT RULES
            ======================

            A new argument MUST:
            - be a **standalone factual or causal statement** (one or two sentences max)
            - NOT restate, paraphrase, quote, or refer to the hypothesis or argument it relates to
            - directly **support** OR directly **attack** the hypothesis or other arguments
            - be included **only** if the relation is explicit, direct, and unambiguous
            - **be taken verbatim from the input text whenever possible**; only minimal trimming or clarification (e.g., replacing pronouns or removing contextual connectors) is allowed. Paraphrasing is forbidden.
            - NOT be highly similar (cosine similarity > 0.9) to any argument in the list of previously extracted arguments

            A new argument MUST NOT:
            - depend on the hypothesis wording
            - contain phrases like “this supports H1”, “this contradicts…”, “unlike the hypothesis…”
            - rely on inference, background knowledge, or indirect association
            - combine multiple unrelated facts
            - be present in the list of previous arguments

            Valid relations:
            - `"support"`: the argument provides a direct factual reason for the target.
            - `"attack"`: the argument directly contradicts or undermines the target.
            If the relation is unclear, indirect, or merely associative → **DO NOT INCLUDE IT**.

            ======================
             INPUT
            ======================

            Input Text:
            \"\"\"{text}\"\"\"

            Hypotheses:
            {json.dumps(hypotheses, indent=2, ensure_ascii=False)}

            Previously extracted arguments:
            {json.dumps(prev_arguments, indent=2, ensure_ascii=False)}

            ======================
             OUTPUT FORMAT
            ======================

            Return **ONLY** a JSON array of objects:

            [
              {{
                "text": "standalone factual argument",
                "relation": "support" or "attack",
                "target": "exact string of the hypothesis or previous argument"
              }}
            ]

            Notes:
            - The `target` field must match exactly a hypothesis or previous argument. Slight modifications will be considered invalid.
            - Avoid generating arguments that are semantically too close to previous ones.
        """


