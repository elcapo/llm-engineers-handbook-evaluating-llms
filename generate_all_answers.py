#!/usr/bin/env python3

import sys
from rich import print
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

from evaluating_llms.answers_dataset import AnswersDataset

if len(sys.argv) < 2:
    print("At least a [cyan]model_id[/cyan] is required to execute this script:\n")
    print("[blue]./generate_all_answers.py [cyan]meta-llama/Meta-Llama-3.1-8B-Instruct")
    exit(1)

if len(sys.argv) > 3:
    print("Only [cyan]model_id[/cyan] and [cyan]endpoint_url[/cyan] are supported parameters:\n")
    print("[blue]./generate_all_answers.py [cyan]mlabonne/TwinLlama-3.1-8B-GGUF https://endpoint-url.location.provider.endpoints.huggingface.cloud")
    exit(1)

model_id = sys.argv[1]
endpoint_url = sys.argv[2] if len(sys.argv) == 3 else None

answers_dataset = AnswersDataset(model_id, endpoint_url)
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