import click

import src.main

@click.command(name='run')
@click.option(
    '-i', '--input-path', 
    type=click.File('r'), 
    required=True,
    help='Path to input .bam or .bed file'
)
@click.option(
    '-c', '--cn-profiles-path',
    type=click.File('r'),
    required=True,
    help='Path to input .bed file with the copy-number profiles for each clone'
)
@click.option(
    '-o', '--output',
    type=click.File('w'),
    required=True,
    help='Path to where the output is written to'
)
@click.option(
    '-m', '--model',
    default='simple',
    help='One of [simple]'
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
@click.option(
    '--progress-bar',
    default=False,
    help='Show progress bar during inference'
)
@click.option(
    '--preprocess',
    default=True,
    help='Preprocess data by removing outliers within CN configurations'
)
def run(**kwargs):
    """ Fit LiquidBayes model to data.
    """
    src.main.run(**kwargs)

@click.group(name='liquid-bayes')
def main():
    pass

main.add_command(run)
