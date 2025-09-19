import os
import json
from datasets import Dataset
import pathlib

from evaluating_llms.answers_generator import AnswersGenerator

class AnswersWriter:
    def __init__(self, model_id: str | None = None, endpoint_url: str | None = None):
        if endpoint_url is not None:
            self.answers_generator = AnswersGenerator(model_id=model_id, endpoint_url=endpoint_url)
        else:
            self.answers_generator = AnswersGenerator(model_id=model_id)

    def get_model_label(self) -> str:
        return self.answers_generator.get_model_label()

    def generate(self):
        for record in self.answers_generator.generate():
            yield record

    def count(self):
        return self.answers_generator.count()

    def get_directory(self) -> str:
        root = pathlib.Path(__file__).parent.parent.resolve()
        directory = os.path.join(root, "datasets/answers")

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