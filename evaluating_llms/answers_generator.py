import os
from dotenv import load_dotenv

import torch
from huggingface_hub import login
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
from transformers import BitsAndBytesConfig

from evaluating_llms.prompts_generator import PromptsGenerator

class AnswersGenerator:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", prompts_generator: PromptsGenerator = PromptsGenerator()):
        load_dotenv()
        login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            device=0 if torch.cuda.is_available() else -1,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 4096,
                "temperature": 0.8,
                "top_p": 0.95,
                "return_full_text": False,
                "clean_up_tokenization_spaces": True,
            },
            model_kwargs={
                "quantization_config": bnb_config,
                "dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }
        )

        self.model = ChatHuggingFace(llm=llm)
        self.prompts_generator = prompts_generator
    
    def generate(self):
        for record in self.prompts_generator.generate():
            answer = self.model.invoke(record["prompt"])
            record["answer"] = answer.content
            yield record

    def count(self):
        return self.prompts_generator.count()

if __name__ == "__main__":
    from rich import box
    from rich.table import Table
    from rich.console import Console

    top = 5
    answers_generator = AnswersGenerator()

    table = Table(title=f"Answers (first {top} records)", show_lines=True, box=box.DOUBLE_EDGE)
    table.add_column("Instruction")
    table.add_column("Output")
    table.add_column("Prompt")
    table.add_column("Answer")

    for record in answers_generator.generate():
        table.add_row(record["instruction"], record["output"], record["prompt"], record["answer"])

        if table.row_count >= top:
            break
    
    console = Console()
    console.print(table)