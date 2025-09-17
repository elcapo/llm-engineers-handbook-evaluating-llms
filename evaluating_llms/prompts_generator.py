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

if __name__ == "__main__":
    from rich import box
    from rich.table import Table
    from rich.console import Console

    top = 5
    prompts_generator = PromptsGenerator()

    table = Table(title=f"Prompts (first {top} records)", show_lines=True, box=box.DOUBLE_EDGE)
    table.add_column("Instruction")
    table.add_column("Output")
    table.add_column("Prompt")

    for record in prompts_generator.generate():
        table.add_row(record["instruction"], record["output"], record["prompt"])

        if table.row_count >= top:
            break
    
    console = Console()
    console.print(table)