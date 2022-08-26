import numpy as np
import pandas as pd
import pysam

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
            "event_id": ":".join([row.contig, str(row.stop), ref, alt]),
            "ref_counts": counts[ref][0],
            "alt_counts": counts[alt][0]
        }

        df.append(out_row)

    return pd.DataFrame(df)


def combine_counts(counts_liquid, counts_clones, verbose):
    """
    Combine counts at SNV positions present across all dataframes into a single dataframe
    Arguments:
        counts_liquid: a pandas dataframe
        counts_clones: a pandas dataframe
    Returns:
        numpy array with shape (L, 2+K) - L=length of intersection of SNV positions across all dfs, K=ref and alt counts for counts_liquid and num clones
    """

    _print("Combining counts at SNV positions from all files", verbose)

    counts_liquid = counts_liquid.drop_duplicates(subset=['event_id']).drop(counts_liquid[counts_liquid.ref_counts + counts_liquid.alt_counts == 0].index)  # drop rows with 0 reads

    # find all SNVs present in all files
    intersection = set(counts_liquid.event_id)
    temp_clones = []
    for clone in counts_clones:
        clone = clone.drop_duplicates(subset=['event_id']).drop(clone[clone.ref_counts + clone.alt_counts == 0].index) # drop rows with 0 reads
        temp_clones.append(clone)
        intersection = intersection & set(clone.event_id)


    # construct an LxJ matrix (L=num common SNVs, J=num clones) approximating mutant copies at SNV l for clone j
    # ASSUME: rows of all dfs are sorted by event_id AND there is at least one read at each SNV (ref_counts + alt_counts > 0)
    counts_liquid_filtered = counts_liquid[counts_liquid['event_id'].isin(intersection)].drop(columns=['event_id']) # filter rows
    counts = [counts_liquid_filtered]
    for clone in temp_clones:
        clone_filtered = clone[clone['event_id'].isin(intersection)] # filter rows
        copy_est = clone_filtered.apply(lambda row: row.alt_counts / (row.ref_counts + row.alt_counts), axis=1) # compute estimate for num mutant copies 
        counts.append(np.reshape(copy_est.to_numpy(), (-1, 1)))
    counts = np.concatenate(counts, axis=1)
    #n = len(counts)
    #counts[:, 2:] = (counts[:, 2:] * (n-1) + .5) / n
    return counts
