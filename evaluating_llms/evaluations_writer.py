import os
import json
import pathlib

from evaluating_llms.answers_evaluator import AnswersEvaluator

class EvaluationsWriter:
    def __init__(self, model_id: str):
        self.answers_evaluator = AnswersEvaluator(model_id)

    def get_model_label(self) -> str:
        return self.answers_evaluator.get_model_label()

    def generate(self):
        for record in self.answers_evaluator.evaluate():
            yield record

    def count(self):
        return self.answers_evaluator.count()

    def get_directory(self) -> str:
        root = pathlib.Path(__file__).parent.parent.resolve()
        directory = os.path.join(root, "datasets/evaluations")

        os.makedirs(directory, exist_ok=True)

        return directory

    def get_jsonl_path(self) -> str:
        directory = self.get_directory()
        model_label = self.get_model_label()

        return os.path.join(directory, f"{model_label}.jsonl")

    def save_evaluations_as_jsonl(self):
        jsonl_path = self.get_jsonl_path()

        for i, item in enumerate(self.generate()):
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(item) + '\n')
            yield item