import os
import string
from functools import reduce
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import pysam
from pyrsistent import m

from src.utils import _print

def get_counts(bam_path, vcf_path, verbose):
    """
    Get reference and alternate allele counts from bam file at SNV positions in vcf file
    Arguments:
        bam_path: a string
        vcf_path: a string
    Returns:
        A pandas dataframe with columns [event_id, ref, alt] - event_id=genomic location, ref=reference allele counts, alt=alternate allele counts
    """

    def check_read(read):
        if read.is_duplicate or read.is_secondary or read.is_qcfail or read.is_supplementary or read.is_unmapped or (read.mapping_quality < 60):
            return False
        else:
            return True

    _print(f"Getting counts from {bam_path} at SNV positions in {vcf_path}", verbose)

    bam = pysam.AlignmentFile(bam_path)
    bcf = pysam.VariantFile(vcf_path)
    df = []

    for row in bcf:
        filters = list(row.filter.keys())

        if (len(filters) > 0) and ("PASS" not in filters):
             continue

        if len(row.alts) > 1:
             continue

        if row.contig[0].isalpha():
            continue

        ref = row.ref

        alt = row.alts[0]

        if (len(ref) > 1) or (len(alt) > 1):
             continue

        counts = dict(
            zip("ACGT", bam.count_coverage(row.contig, row.start, row.stop, quality_threshold=20, read_callback=check_read))
        )

        out_row = {
            "event_id": ":".join([row.contig, str(row.stop)]),
            "ref_counts": counts[ref][0],
            "alt_counts": counts[alt][0]
        }

        df.append(out_row)

    return pd.DataFrame(df)

def process_counts(_counts_liquid, _counts_clones, _cn_profiles, verbose):
    """
    Get ref and alt counts from liquid biopsy and estimates for mutant copies for each clone
    Arguments:
        counts_liquid: a pandas dataframe
        counts_clones: list of pandas dataframes
        cn_profiles: a numpy array
    Returns:
        ndarray with shape (L, 2+K) - L=length of intersection of SNV positions across all dfs, 2+K=ref and alt counts for liquid biopsy and K estimates for mutant copies for each clone (excluding normal)
    """

    def valid(event_id, _ranges):
        """
        Check if the genomic position described by event_id is included in bins defined by ranges
        Arguments:
            event_id: a string
            _ranges: a dictionary
        Returns:
            boolean value indicating whether the genomic position described by event_id is included in bins defined by ranges
        """
        event_id_list = event_id.split(':') 
        chromosome, pos = int(event_id_list[0]), int(event_id_list[1])
        return _ranges[chromosome][0] <= pos <= _ranges[chromosome][1]    

    def drop_duplicates_and_0_reads(counts_df):
        return counts_df.drop_duplicates(subset=['event_id']).drop(counts_df[counts_df.ref_counts + counts_df.alt_counts == 0].index)

    def compute_vaf(row):
        return row['alt_counts'] / (row['ref_counts'] + row['alt_counts'])

    def get_chr(event_id):
        """
        Given an event_id (ex. 1:3005513), return chromosome component
        Arguments:
            event_id: a pandas Series
        Returns:
            string representing chromosome
        """
        return event_id.split(':')[0]

    def get_pos(event_id):
        """
        Given an event_id (ex. 1:3005513), return position component
        Arguments:
            event_id: a pandas Series
        Returns:
            string representing position
        """
        return event_id.split(':')[1]

    _print("Processing counts from liquid biopsy and estimating mutant copies for each clone", verbose)

    # load CN profiles
    cn_profiles = pd.DataFrame(data=_cn_profiles)

    # get genomic intervals spanned by CN profile bins for each chromosome
    ranges = m()
    for chromosome in cn_profiles.iloc[:, 0].unique().astype(int):
        subset = cn_profiles.loc[cn_profiles.iloc[:, 0].astype(int) == chromosome]
        ranges = ranges.set(chromosome, (int(subset.iloc[0, 1]), int(subset.iloc[-1, 2])))  # store min and max in tuple

    # drop duplicates and rows with 0 reads
    counts_liquid = drop_duplicates_and_0_reads(_counts_liquid)
    counts_clones = [drop_duplicates_and_0_reads(_counts_clone) for _counts_clone in _counts_clones]

    # remove SNVs outside of genomic intervals spanned by CN profile bins
    counts_liquid = counts_liquid.loc[counts_liquid['event_id'].apply(valid, args=(ranges,))]
    counts_clones = [counts_clone.loc[counts_clone['event_id'].apply(valid, args=(ranges,))] for counts_clone in counts_clones]
    
    # construct a df where keys are chr:pos and vals are a dict containing clone VAFs
    # ex.
    # chr:pos    A    B    C
    # 1:77655    .8   .4   0
    #   ...     ...  ...  ...
    # 22:4932    .4   .25  .5

    # compute VAF for each clone and store in column
    uppercase_letters = list(string.ascii_uppercase) # for naming clones
    for i, counts_clone in enumerate(counts_clones):
        counts_clone[uppercase_letters[i]] = counts_clone.apply(compute_vaf, axis=1)
        counts_clone.drop(columns=['ref_counts', 'alt_counts'], inplace=True) # drop ref and alt counts

    # outer join all clone vafs
    vafs = reduce(lambda left, right: pd.merge(left, right, on=['event_id'], how='outer'), counts_clones).fillna(0)
    counts_liquid_and_vafs = pd.merge(counts_liquid, vafs, on=['event_id'], how='inner') # only keep positions that are present in both

    # simply return empty df if there are no common SNVs between liquid and tissue biopsies
    if counts_liquid_and_vafs.empty:
        return None

    # get corresponding clone CN values at SNV positions
    cn_row = 0
    cns = []
    for i in range(counts_liquid_and_vafs.shape[0]):
        # iterate until we get to the right chromosome
        while int(get_chr(counts_liquid_and_vafs['event_id'].iloc[i])) != int(cn_profiles.iloc[cn_row, 0]):
            cn_row += 1
        # iterate until we get to the right position
        while int(get_pos(counts_liquid_and_vafs['event_id'].iloc[i])) > cn_profiles.iloc[cn_row, 2]:
            cn_row += 1
        cns.append(cn_profiles.iloc[cn_row, 3:cn_profiles.shape[1]-1])  # don't need normal so omit last column
    cns = np.array(cns)

    mutant_copies = counts_liquid_and_vafs.drop(columns=['event_id', 'ref_counts', 'alt_counts']).to_numpy() * cns

    # construct an Lx(2+K) matrix (L=num SNVs post filtering, 2+K=ref & alt counts for liquid biopsy + mutant copies estimates at SNV l for each clone j
    res = np.c_[counts_liquid_and_vafs[['ref_counts', 'alt_counts']].to_numpy(), mutant_copies]
    res = res[~np.all(res[:, 2:] == 0, axis=1)]
    return res
