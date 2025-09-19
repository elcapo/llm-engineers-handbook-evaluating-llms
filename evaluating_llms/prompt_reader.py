import os

class PromptReader:
    def read_prompt(self, prompt_name: str) -> str:
        prompt_path = os.path.abspath(__file__+f"/../../prompts/{prompt_name}.md")

        with open(prompt_path, "r") as f:
            prompt = f.read()

        return prompt