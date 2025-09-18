#!/usr/bin/env python3

import sys
from rich import box, print
from rich.table import Table
from rich.console import Console

from evaluating_llms.answers_generator import AnswersGenerator

if len(sys.argv) < 2:
    print("At least a [cyan]model_id[/cyan] is required to execute this script:\n")
    print("[blue]./preview_answers.py [cyan]meta-llama/Meta-Llama-3.1-8B-Instruct")
    exit(1)

if len(sys.argv) > 3:
    print("Only [cyan]model_id[/cyan] and [cyan]endpoint_url[/cyan] are supported parameters:\n")
    print("[blue]./preview_answers.py [cyan]mlabonne/TwinLlama-3.1-8B-GGUF https://endpoint-url.location.provider.endpoints.huggingface.cloud")
    exit(1)

model_id = sys.argv[1]
endpoint_url = sys.argv[2] if len(sys.argv) == 3 else None

top = 5
answers_generator = AnswersGenerator(model_id=model_id, endpoint_url=endpoint_url)

table = Table(title=f"Answers (model: {answers_generator.get_model_label()}, first: {top} records)", show_lines=True, box=box.DOUBLE_EDGE)
table.add_column("Instruction")
table.add_column("Output")
table.add_column("Prompt")
table.add_column("Answer", style="cyan")

for record in answers_generator.generate():
    table.add_row(record["instruction"], record["output"], record["prompt"], record["answer"])

    if table.row_count >= top:
        break

console = Console()
console.print(table)