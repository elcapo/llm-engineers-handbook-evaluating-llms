import os
from datasets import Dataset
import pathlib

class AnswersReader:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id

    def get_model_label(self) -> str:
        return self.model_id.split("/")[-1].lower()

    def generate(self):
        for record in self.load_jsonl():
            yield record

    def count(self):
        records = self.load_jsonl()
        return len(records)

    def get_directory(self) -> str:
        root = pathlib.Path(__file__).parent.parent.resolve()
        directory = os.path.join(root, "datasets/answers")

        os.makedirs(directory, exist_ok=True)

        return directory

    def get_jsonl_path(self) -> str:
        directory = self.get_directory()
        model_label = self.get_model_label()

        return os.path.join(directory, f"{model_label}.jsonl")

    def load_jsonl(self):
        jsonl_path = self.get_jsonl_path()

        return Dataset.from_json(jsonl_path)