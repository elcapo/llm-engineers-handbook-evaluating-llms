#!/usr/bin/env python3

from rich import box
from rich.table import Table
from rich.console import Console
from evaluating_llms.prompts_generator import PromptsGenerator

top = 5
prompts_generator = PromptsGenerator()

table = Table(title=f"Prompts (first {top} records)", show_lines=True, box=box.DOUBLE_EDGE)
table.add_column("Instruction")
table.add_column("Output", style="red")
table.add_column("Prompt", style="cyan")

for record in prompts_generator.generate():
    table.add_row(record["instruction"], record["output"], record["prompt"])

    if table.row_count >= top:
        break

console = Console()
console.print(table)