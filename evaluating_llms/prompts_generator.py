from evaluating_llms.instructions_dataset import InstructionsDataset

PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""

class PromptsGenerator:
    def __init__(self, instructions_dataset: InstructionsDataset = InstructionsDataset()):
        self.instructions_dataset = instructions_dataset

    def generate(self):
        for record in self.instructions_dataset.get_record():
            instruction = record["instruction"]
            record["prompt"] = PROMPT_TEMPLATE.format(instruction)
            yield record

    def count(self):
        return self.instructions_dataset.count()