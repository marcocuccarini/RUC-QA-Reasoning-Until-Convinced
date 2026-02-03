import os
import uuid
import datetime
from pathlib import Path
import time
import ollama

# ===========================
# LLM Response & Role Classes
# ===========================
class ResponseType:
    GENERATED = "generated"
    ERROR = "error"

class LLMResponse:
    def __init__(self, prompt_id, raw_text, timestamp, response_type):
        self.prompt_id = prompt_id
        self.raw_text = raw_text
        self.timestamp = timestamp
        self.response_type = response_type

# ===========================
# Local Ollama Server
# ===========================
class OllamaServer:
    """
    Local Ollama server wrapper.
    """
    def __init__(self):
        self.client = ollama  # uses the local ollama Python SDK

    def get_models_list(self):
        # Local Ollama client doesn't have list API like remote
        return ["llama2", "my_local_model"]  # replace with models you have locally

    def download_model_if_not_exists(self, model_name):
        # For local Ollama, models are already available; just print
        print(f"Using local model: {model_name}")

# ===========================
# Local Ollama Chat Class
# ===========================
class OllamaChat:
    USER = "user"
    ASSISTANT = "assistant"

    def __init__(self, server: OllamaServer, model: str):
        self.server = server
        self.model = model
        self.messages = []
        self.server.download_model_if_not_exists(model)

    def add_history(self, content: str, role: str):
        self.messages.append({"role": role, "content": content})

    def clear_history(self):
        self.messages = []

    def send_prompt(self, prompt: str, prompt_uuid: str = None, use_history=False, stream=False, max_retries=3):
        if prompt_uuid is None:
            prompt_uuid = str(uuid.uuid4())

            # If use_history=True self.messages + [{"role": self.USER, "content": prompt}]
            # If use_history=False [{"role": self.USER, "content": prompt}]

        messages = self.messages + [{"role": self.USER, "content": prompt}] if use_history else [{"role": self.USER, "content": prompt}]

        retries = 0
        while retries < max_retries:
            try:
                # Local Ollama Python SDK
                response = ollama.chat(model=self.model, messages=messages, stream=stream)

                complete_message = ""
                if stream:
                    for line in response:
                        complete_message += line["message"]["content"]
                        print(line["message"]["content"], end="", flush=True)
                else:
                    complete_message = response.get("message", {}).get("content", "").strip()

                if use_history:
                    self.add_history(prompt, self.USER)
                    self.add_history(complete_message, self.ASSISTANT)

                return LLMResponse(
                    prompt_id=prompt_uuid,
                    raw_text=complete_message,
                    timestamp=datetime.datetime.now(),
                    response_type=ResponseType.GENERATED
                )

            except Exception as e:
                retries += 1
                print(f"\n⚠️ Error sending prompt (attempt {retries}/{max_retries}): {e}")
                time.sleep(5)
                if retries >= max_retries:
                    print("❌ Maximum retries reached. Model seems disconnected.")
                    input("Fix the connection and press ENTER to continue...")
                    retries = 0