# LiquidBayes
LiquidBayes is a Bayesian Network (BN) for inferring tumour fraction and clonal prevalences from whole genome sequencing of cell-free DNA (cfDNA) and Direct Library Preparation (DLP+) of a matched tissue biopsy. 

## Installation
Using pip:
```
pip install git+https://git@github.com/Roth-Lab/LiquidBayes.git
```

## Usage
LiquidBayes offers 2 models: `cn` and `cn_snv`. The required parameters will differ depending on which model is used.

### `cn` model
#### Required Parameters
- `-i --input-path` Path to input .bam
- `-c --cn-profiles-path` Path to .bed file with the copy-number profiles for each clone
- `-o --output` Write output to this file
- `--gc` Path to gc content .wig file
- `--mapp` Path to the mappability .wig file

### `cn_snv` model
#### Required Parameters
All parameters in `cn` model **plus**:
- `-l --liquid-vcf` Path to liquid biopsy .vcf file
- `-b --tissue-bams` Path to clone .bam files (ex. ... -t path_to_clone_1 -t path_to_clone_2 -t path_to_clone_3 ...) - order of clones on the command line must be the same as in copy-number profiles (--cn-profiles-path)
- `-t --tissue-vcfs` Path to clone .vcf files (ex. ... -t path_to_clone_1 -t path_to_clone_2 -t path_to_clone_3 ...) - order of clones on the command line must be the same as in copy-number profiles (--cn-profiles-path)


Simply run `liquid-bayes` with the corresponding parameters in the command line!
