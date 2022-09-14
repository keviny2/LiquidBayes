# LiquidBayes
LiquidBayes is a Bayesian Network (BN) for inferring tumour fraction and clonal prevalences from whole genome sequencing of cell-free DNA (cfDNA) and Direct Library Preparation (DLP+) of a matched tissue biopsy. 

## Installation
Using pip:
```
pip install git+https://git@github.com/Roth-Lab/LiquidBayes.git
```

## Usage
LiquidBayes offers 2 models: `cn` and `cn_snv`. The required parameters will differ depending on which model is used. Refer to help page by typing `liquid-bayes --help` in the terminal.

### `cn` model
#### Required Parameters
- `-i --input-path` Path to input bam file
- `-c --cn-profiles-path` Path to file with the copy-number profiles for each clone (.bed format)
- `-o --output` Write output to this file (.csv format)
- `--gc` Path to gc content (.wig format)
- `--mapp` Path to the mappability (.wig format)

#### Example
`liquid-bayes -i input.bam --gc hg38.gc.wig --mapp hg38.map.wig -c cn_profiles.bed -o results.csv -m cn_snv -n 2000 -w 200 -s 1 --progress-bar True --verbose True`

### `cn_snv` model
#### Required Parameters
All parameters in `cn` model **plus**:
- `-l --liquid-vcf` Path to liquid biopsy .vcf file (can be compressed in .gz file)
- `-b --tissue-bams` Path to clone .bam files (ex. ... -t path_to_clone_1 -t path_to_clone_2 -t path_to_clone_3 ...) - order of clones on the command line must be the same as in copy-number profiles (--cn-profiles-path)
- `-t --tissue-vcfs` Path to clone .vcf files (ex. ... -t path_to_clone_1 -t path_to_clone_2 -t path_to_clone_3 ...) - order of clones on the command line must be the same as in copy-number profiles (--cn-profiles-path)

#### Example
`liquid-bayes -i input.bam --gc hg38.gc.wig --mapp hg38.map.wig -c cn_profiles.bed -o results.csv -l liquid.vcf.gz -b A.bam -b B.bam -t A.vcf.gz -t B.vcf.gz -m cn_snv -n 2000 -w 200 -s 1 --progress-bar True --verbose True`
