#!/usr/bin/env python3

from rich import box
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table

import numpy as np
from datasets import Dataset

reference_evaluations = Dataset.from_json("datasets/evaluations/meta-llama-3.1-8b-instruct.jsonl")
finetuned_evaluations = Dataset.from_json("datasets/evaluations/twinllama-3.1-8b-gguf.jsonl")
dpo_evaluations = Dataset.from_json("datasets/evaluations/twinllama-3.1-8b-dpo-gguf.jsonl")

count = len(reference_evaluations) + len(finetuned_evaluations) + len(dpo_evaluations)

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
    task = progress.add_task(f"Computing evaluations", total=count, start=True)

    reference_scores = { "accuracy": [], "style": []}
    for record in reference_evaluations:
        reference_scores["accuracy"].append(record["evaluation"]["accuracy"]["score"])
        reference_scores["style"].append(record["evaluation"]["style"]["score"])

        progress.update(task, advance=1)
        progress.refresh()

    finetuned_scores = { "accuracy": [], "style": []}
    for record in finetuned_evaluations:
        finetuned_scores["accuracy"].append(record["evaluation"]["accuracy"]["score"])
        finetuned_scores["style"].append(record["evaluation"]["style"]["score"])

        progress.update(task, advance=1)
        progress.refresh()

    dpo_scores = { "accuracy": [], "style": []}
    for record in dpo_evaluations:
        dpo_scores["accuracy"].append(record["evaluation"]["accuracy"]["score"])
        dpo_scores["style"].append(record["evaluation"]["style"]["score"])

        progress.update(task, advance=1)
        progress.refresh()

    table = Table(title="Evaluation Summary", show_lines=True, box=box.DOUBLE_EDGE)
    table.add_column("Model")
    table.add_column("[green]Accuracy[/green] (avg)", justify="right", style="green")
    table.add_column("[green]Accuracy[/green] (std)", justify="right", style="green")
    table.add_column("[cyan]Style[/cyan] (avg)", justify="right", style="cyan")
    table.add_column("[cyan]Style[/cyan] (std)", justify="right", style="cyan")

    table.add_row(
        "Reference",
        "{:.2f}".format(np.mean(reference_scores["accuracy"])),
        "{:.2f}".format(np.std(reference_scores["accuracy"])),
        "{:.2f}".format(np.mean(reference_scores["style"])),
        "{:.2f}".format(np.std(reference_scores["style"])),
    )

    table.add_row(
        "Finetuned",
        "{:.2f}".format(np.mean(finetuned_scores["accuracy"])),
        "{:.2f}".format(np.std(finetuned_scores["accuracy"])),
        "{:.2f}".format(np.mean(finetuned_scores["style"])),
        "{:.2f}".format(np.std(finetuned_scores["style"])),
    )

    table.add_row(
        "DPO",
        "{:.2f}".format(np.mean(dpo_scores["accuracy"])),
        "{:.2f}".format(np.std(dpo_scores["accuracy"])),
        "{:.2f}".format(np.mean(dpo_scores["style"])),
        "{:.2f}".format(np.std(dpo_scores["style"])),
    )

    console = Console()
    console.print(table)