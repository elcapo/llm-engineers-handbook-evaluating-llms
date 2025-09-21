#!/usr/bin/env python3

from rich import box
from rich.table import Table
from rich.console import Console
from evaluating_llms.instructions_dataset import InstructionsDataset

top = 5
dataset = InstructionsDataset()

table = Table(title=f"Instructions (first {top} records)", show_lines=True, box=box.DOUBLE_EDGE)
table.add_column("Instruction", style="cyan")
table.add_column("Output", style="red")

for record in dataset.get_record():
    table.add_row(record["instruction"], record["output"])

    if table.row_count >= top:
        break

console = Console()
console.print(table)