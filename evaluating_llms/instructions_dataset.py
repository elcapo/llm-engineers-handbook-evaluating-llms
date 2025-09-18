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