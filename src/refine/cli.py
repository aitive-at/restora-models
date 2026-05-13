import typer

app = typer.Typer(help="refine — multi-task image restoration", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """refine CLI."""


@app.command()
def version() -> None:
    from refine import __version__
    typer.echo(__version__)
