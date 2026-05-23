import typer
from rich.console import Console
from rich.table import Table

from iltools.datasets.loaders import load_dataset_loader, registered_dataset_loaders

app = typer.Typer(help="Imitation Learning Tools CLI")
console = Console()


@app.command()
def load(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to load"),
    data_path: str = typer.Option(None, help="Path to the dataset"),
    model_path: str = typer.Option(None, help="Path to the SMPL-X model (for AMASS)"),
    control_freq: int = typer.Option(
        30, help="Control frequency for LocoMuJoCo datasets"
    ),
):
    """
    Loads a dataset and prints its metadata.
    """
    with console.status(f"[bold green]Loading {dataset_name}...[/bold green]"):
        try:
            loader_cls = load_dataset_loader(dataset_name)
        except KeyError:
            choices = ", ".join(registered_dataset_loaders())
            console.print(
                "[bold red]"
                f"Unknown dataset: {dataset_name}. Choices: {choices}"
                "[/bold red]"
            )
            raise typer.Exit(1) from None
        except ImportError as exc:
            raise typer.BadParameter(str(exc)) from exc

        normalized_name = dataset_name.strip().lower().replace("-", "_")
        if normalized_name == "loco_mujoco":
            loader = loader_cls(
                env_name="Humanoid", task="walk", control_freq=control_freq
            )
        elif normalized_name in {"lafan1", "lafan1_csv"}:
            loader = loader_cls(data_path)
        else:
            loader = loader_cls(data_path)

        num_trajectories = len(loader)
        metadata = loader.metadata

    table = Table(title=f"{metadata.name} Metadata")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")
    for field, value in metadata.dict().items():
        table.add_row(field, str(value))

    console.print(table)
    console.print(f"Loaded {num_trajectories} trajectories.")


@app.command()
def retarget():
    """
    Retargets a trajectory to a new robot (placeholder).
    """
    console.print(
        "[bold yellow]This is a placeholder for the retargeting command.[/bold yellow]"
    )


if __name__ == "__main__":
    app()
