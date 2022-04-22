import click

import src.main

@click.command()
@click.option(
    '-m', '--model',
    default='simple',
    help='one of [simple]'
)
@click.option(
    '-n', '--num-samples',
    default=3000,
    help='Number of samples to draw'
)
@click.option(
    '-w', '--num-warmup',
    default=100,
    help='Number of warm up samples to draw'
)
@click.option(
    '-s', '--seed',
    default=1,
    help='Seed for random functions'
)
@click.argument(
    'data', 
    type=click.File('r'), 
    required=True
)
@click.argument(
    'cn-profiles',
    type=click.File('r'),
    required=True
)
@click.argument(
    'output',
    type=click.File('w'),
    required=True
)
def run(*kwargs)
    """ Fit LiquidBayes model to data.
    """
    src.main.run(**kwargs)
