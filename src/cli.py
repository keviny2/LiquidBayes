import click

import src.main

@click.command(name='run')
@click.option(
    '-i', '--input-path', 
    type=click.STRING,
    required=True,
    help='Path to input .bam'
)
@click.option(
    '-c', '--cn-profiles-path',
    type=click.STRING,
    required=True,
    help='Path to input .bed file with the copy-number profiles for each clone'
)
@click.option(
    '-o', '--output',
    type=click.STRING,
    required=True,
    help='Path to where the output is written to'
)
@click.option(
    '-l', '--liquid-vcf',
    type=click.STRING,
    default='',
    help='Path to liquid biopsy vcf file'
)
@click.option(
    '-b', '--tissue-bams',
    type=click.STRING,
    default=[''],
    help='Path to clone bam files (ex. ... -t path_to_clone_1 -t path_to_clone_2 -t path_to_clone_3 ...) - order of clones on the command line must be the same as copy-number profiles (--cn-profiles-path)',
    multiple=True
)
@click.option(
    '-t', '--tissue-vcfs',
    type=click.STRING,
    default=[''],
    help='Path to clone vcf files (ex. ... -t path_to_clone_1 -t path_to_clone_2 -t path_to_clone_3 ...) - order of clones on the command line must be the same as copy-number profiles (--cn-profiles-path)',
    multiple=True
)   
@click.option(
    '-m', '--model',
    type=click.STRING,
    default='cn',
    help='One of [cn, one-more-clone]'
)
@click.option(
    '-n', '--num-samples',
    type=click.INT,
    default=3000,
    help='Number of samples to draw'
)
@click.option(
    '-w', '--num-warmup',
    type=click.INT,
    default=100,
    help='Number of warm up samples to draw'
)
@click.option(
    '-s', '--seed',
    type=click.INT,
    default=1,
    help='Seed for random functions'
)
@click.option(
    '--gc',
    type=click.STRING,
    required=True,
    help='Path to the gc content wig file'
)
@click.option(
    '--mapp',
    type=click.STRING,
    required=True,
    help='Path to the mappability wig file'
)
@click.option(
    '--progress-bar',
    type=click.BOOL,
    default=False,
    help='Show progress bar during inference'
)
@click.option(
    '--chrs',
    type=click.STRING,
    default='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22',
    help='Chromosomes present in bam file'
)
@click.option(
    '--bin-size',
    type=click.STRING,
    default='500000',
    help='Bin size for computing read counts'
)
@click.option(
    '--qual',
    type=click.STRING,
    default='20',
    help='Specify the mapping quality value below which reads are ignored'
)
@click.option(
    '--verbose',
    type=click.BOOL,
    default=False,
    help='Allow printing'
)
def run(**kwargs):
    """ Fit LiquidBayes model to data.
    """
    src.main.run(**kwargs)

@click.group(name='liquid-bayes')
def main():
    pass

main.add_command(run)
