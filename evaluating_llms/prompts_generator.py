from evaluating_llms.instructions_dataset import InstructionsDataset
from evaluating_llms.prompt_reader import PromptReader

class PromptsGenerator:
    def __init__(self, instructions_dataset: InstructionsDataset = InstructionsDataset()):
        self.instructions_dataset = instructions_dataset

    def get_prompt(self) -> str:
        return PromptReader().read_prompt("generate-answer")

    def generate(self):
        for record in self.instructions_dataset.get_record():
            instruction = record["instruction"]
            prompt = self.get_prompt()
            record["prompt"] = prompt.format(instruction)
            yield record

    def count(self):
        return self.instructions_dataset.count()