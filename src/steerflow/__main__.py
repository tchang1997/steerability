import click
from steerflow.app import create_app

@click.group()
def cli():
    """SteerFlow CLI entry point."""
    pass

@cli.command()
@click.option("--port", default=5000)
def launch(port):
    """Launch the interactive flow UI."""
    app = create_app()
    app.run(debug=True, port=port)

if __name__ == "__main__":
    cli()
