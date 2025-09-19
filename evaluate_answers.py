#!/usr/bin/env python3

import sys
from rich import print
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

from evaluating_llms.evaluations_writer import EvaluationsWriter

if len(sys.argv) != 2:
    print("At least a [cyan]model_id[/cyan] is required to execute this script:\n")
    print("[blue]./evaluate_answers.py [cyan]meta-llama/Meta-Llama-3.1-8B-Instruct")
    exit(1)

model_id = sys.argv[1]

evaluations_writer = EvaluationsWriter(model_id)
count = evaluations_writer.count()
model_label = evaluations_writer.get_model_label()

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
    task = progress.add_task(f"Evaluating answers for [bold]{model_label}[/bold]", total=count, start=True)

    for record in evaluations_writer.save_evaluations_as_jsonl():
        progress.update(task, advance=1)
        progress.refresh()