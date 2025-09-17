from datasets import load_dataset

# The default datasource is available at:
# - https://huggingface.co/datasets/mlabonne/llmtwin/

class InstructionsDataset:
    def __init__(self, dataset_name: str = "mlabonne/llmtwin", split: str = "test"):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)

    def get_record(self):
        for record in self.dataset:
            yield record

    def count(self):
        return sum(1 for _ in self.dataset)

if __name__ == "__main__":
    from rich import box
    from rich.table import Table
    from rich.console import Console

    top = 5
    dataset = InstructionsDataset()

    table = Table(title=f"Instructions (first {top} records)", show_lines=True, box=box.DOUBLE_EDGE)
    table.add_column("Instruction")
    table.add_column("Output")

    for record in dataset.get_record():
        table.add_row(record["instruction"], record["output"])

        if table.row_count >= top:
            break
    
    console = Console()
    console.print(table)