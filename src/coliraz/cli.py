import typer

app = typer.Typer(help="coliraz — image colorization", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """coliraz CLI."""


@app.command()
def version() -> None:
    from coliraz import __version__
    typer.echo(__version__)
