from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from evaluating_llms.answers_reader import AnswersReader
from evaluating_llms.prompt_reader import PromptReader

class Score(BaseModel):
    """Score and analysis that justifies the score"""
    score: int = Field(description="Score from 1 to 3 of the evaluation")
    analysis: str = Field(description="Reason why the score was given")

class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""
    accuracy: Score = Field(description="Score for the accuracy evaluation")
    style: Score = Field(description="Score for the accuracy evaluation")

class AnswersEvaluator:
    def __init__(self, model_id: str):
        load_dotenv()

        self.answers_reader = AnswersReader(model_id)

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.8, max_tokens=1000)
        self.model = model.with_structured_output(ResponseFormatter)

    def count(self) -> int:
        return self.answers_reader.count()

    def get_model_label(self) -> str:
        return self.answers_reader.get_model_label()

    def get_system_prompt(self) -> str:
        return PromptReader().read_prompt("evaluator-system")

    def get_task_prompt(self) -> str:
        return PromptReader().read_prompt("evaluate-answer")

    def evaluate(self, batch_size=4):
        batch = []
        for record in self.answers_reader.generate():
            batch.append(record)

            if len(batch) >= batch_size:
                yield from self.process_batch(batch)

                batch = []
        
        yield from self.process_batch(batch)

    def process_batch(self, batch):
        prompt_template = ChatPromptTemplate([
            ("system", self.get_system_prompt()),
            ("user", self.get_task_prompt())
        ])

        prompts = [prompt_template.invoke({"instruction": record["instruction"], "answer": record["answer"]}) for record in batch]

        evaluations = self.model.batch(prompts)

        for record, evaluation in zip(batch, evaluations):
            record["evaluation"] = {
                "accuracy": {
                    "score": evaluation.accuracy.score,
                    "analysis": evaluation.accuracy.analysis,
                },
                "style": {
                    "score": evaluation.style.score,
                    "analysis": evaluation.style.analysis,
                }
            }

            yield record