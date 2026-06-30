import sys
from pathlib import Path

import click

from src.report import generate_report
from src.runner import run_suite


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--suite", required=True, help="Path to the YAML test suite file.")
@click.option("--models", required=True, help="Comma-separated list of model names.")
@click.option("--output", default="reports", show_default=True, help="Output directory for reports.")
def run(suite: str, models: str, output: str) -> None:
    suite_path = Path(suite)
    if not suite_path.exists():
        click.echo(f"Error: suite file not found: {suite}", err=True)
        sys.exit(1)

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    suite_name = suite_path.name

    click.echo(f"Running eval suite: {suite_name} against {', '.join(model_list)}...")

    results = run_suite(suite_path, model_list)
    generate_report(results, output_dir=output)

    click.echo(f"Done. Report saved to {output}/")


if __name__ == "__main__":
    cli()
