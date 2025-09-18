from dotenv import load_dotenv

from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

from evaluating_llms.prompts_generator import PromptsGenerator

class AnswersGenerator:
    def __init__(self, model_id: str | None = None, endpoint_url: str | None = None, prompts_generator: PromptsGenerator = PromptsGenerator()):
        load_dotenv()

        model_kwargs = {
            "task": "text-generation",
            "max_new_tokens": 4098,
            "temperature": 0.8,
            "top_p": 0.95,
            "return_full_text": False,
        }

        if endpoint_url is not None:
            model_kwargs["endpoint_url"] = endpoint_url
        else:
            model_kwargs["repo_id"] = model_id

        llm = HuggingFaceEndpoint(**model_kwargs)

        self.model_id = model_id
        self.model = ChatHuggingFace(llm=llm)
        self.prompts_generator = prompts_generator

    def get_model_label(self) -> str:
        return self.model_id.split("/")[-1].lower()

    def generate(self, batch_size=4):
        batch = []
        for record in self.prompts_generator.generate():
            batch.append(record)

            if len(batch) >= batch_size:
                yield from self.process_batch(batch)

                batch = []
        
        yield from self.process_batch(batch)

    def process_batch(self, batch):
        prompts = [record["prompt"] for record in batch]

        answers = self.model.batch(prompts)

        for record, answer in zip(batch, answers):
            record["answer"] = answer.content
            yield record