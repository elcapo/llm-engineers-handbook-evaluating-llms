import os
import json
from datasets import Dataset
import pathlib

from evaluating_llms.answers_generator import AnswersGenerator

class AnswersDataset:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_id = model_id
        self.answers_generator = AnswersGenerator(model_id)

    def get_model_label(self) -> str:
        return self.model_id.split("/")[-1].lower()

    def generate(self):
        for record in self.answers_generator.generate():
            yield record

    def count(self):
        return self.answers_generator.count()

    def get_directory(self) -> str:
        root = pathlib.Path(__file__).parent.parent.resolve()
        directory = os.path.join(root, "datasets")

        os.makedirs(directory, exist_ok=True)

        return directory

    def get_jsonl_path(self) -> str:
        directory = self.get_directory()
        model_label = self.get_model_label()

        return os.path.join(directory, f"{model_label}.jsonl")

    def save_answers_as_jsonl(self):
        jsonl_path = self.get_jsonl_path()

        for i, item in enumerate(self.generate()):
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(item) + '\n')
            yield item

    def load_jsonl(self):
        jsonl_path = self.get_jsonl_path()

        return Dataset.from_json(jsonl_path)

def store_answers_for_model(model_id: str):
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.console import Console

    answers_dataset = AnswersDataset(model_id)
    count = answers_dataset.count()
    model_label = answers_dataset.get_model_label()

    console = Console()
    console.clear()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Generating answers for [bold]{model_label}[/bold]", total=count, start=True)

        for record in answers_dataset.save_answers_as_jsonl():
            progress.update(task, advance=1)
            progress.refresh()

if __name__ == "__main__":
    store_answers_for_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
    store_answers_for_model("mlabonne/TwinLlama-3.1-8B")
    store_answers_for_model("mlabonne/TwinLlama-3.1-8B-DPO")