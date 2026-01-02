import typer
from rich.console import Console
from rich.table import Table

from iltools.datasets.amass.loader import AmassLoader
from iltools.datasets.loco_mujoco.loader import LocoMuJoCoLoader
from iltools.datasets.trajopt.loader import TrajoptLoader

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
        if dataset_name == "amass":
            loader = AmassLoader(data_path, model_path)
        elif dataset_name == "loco_mujoco":
            loader = LocoMuJoCoLoader(
                env_name="Humanoid", task="walk", control_freq=control_freq
            )
        elif dataset_name == "trajopt":
            loader = TrajoptLoader(data_path)
        else:
            console.print(f"[bold red]Unknown dataset: {dataset_name}[/bold red]")
            raise typer.Exit(1)

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
