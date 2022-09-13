import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
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


def combine_counts(counts_liquid, counts_clones, cn_profiles, verbose):
    """
    Get ref and alt counts from liquid biopsy and estimates for mutant copies for each clone
    Arguments:
        counts_liquid: a pandas dataframe
        counts_clones: a pandas dataframe
        cn_profiles: a numpy array
    Returns:
        numpy array with shape (L, 2+K) - L=length of intersection of SNV positions across all dfs, 2+K=ref and alt counts for liquid biopsy and K estimates for mutant copies for each clone (excluding normal)
    """
    
    def valid(event_id, ranges):
        """
        Check if the genomic position described by event_id is included in bins defined by ranges
        Arguments:
            event_id: a string
            ranges: a dictionary
        Returns:
            boolean value indicating whether the genomic position described by event_id is included in bins defined by ranges
        """
        event_id_list = event_id.split(':') 
        chromosome, pos = int(event_id_list[0]), int(event_id_list[1])
        return ranges[chromosome][0] <= pos <= ranges[chromosome][1]    

    _print("Combining counts at SNV positions from all files", verbose)

    cn_profiles = pd.DataFrame(data=cn_profiles)
    counts_liquid = counts_liquid.drop_duplicates(subset=['event_id']).drop(counts_liquid[counts_liquid.ref_counts + counts_liquid.alt_counts == 0].index)  # drop rows with 0 reads

    # construct dict containing the max and min among bins for each chromosome in cn_profiles
    ranges = {}
    for chromosome in cn_profiles.iloc[:, 0].unique().astype(int):
        subset = cn_profiles.loc[cn_profiles.iloc[:, 0].astype(int) == chromosome]
        ranges[chromosome] = (int(subset.iloc[0, 1]), int(subset.iloc[-1, 2]))  # store min and max in tuple

    # remove events that are not included in cn_profiles bins
    counts_liquid = counts_liquid.loc[counts_liquid['event_id'].apply(valid, args=(ranges,))]
    
    # find all SNVs present in all files
    intersection = set(counts_liquid['event_id'])
    temp_clones = []
    for clone in counts_clones:
        clone = clone.drop_duplicates(subset=['event_id']).drop(clone[clone.ref_counts + clone.alt_counts == 0].index) # drop rows with 0 reads
        temp_clones.append(clone)
        intersection = intersection & set(clone.event_id)   
        
    # get all bins (rows) from cn_profiles that contain an SNV shared across all clones
    counts_liquid_filtered = counts_liquid[counts_liquid['event_id'].isin(intersection)]
    counts_liquid_filtered[['chr', 'pos']] = counts_liquid_filtered['event_id'].str.split(':', expand=True).iloc[:, :2]
    counts_liquid_filtered.drop(columns=['event_id'], inplace=True) 
    
    # REQUIRES: all snv pos are in a cn_profiles bin
    # construct a pandas dataframe with CN values for SNV positions
    cn_row = 0
    res = []
    for i in range(counts_liquid_filtered.shape[0]):
        while int(counts_liquid_filtered['chr'].iloc[i]) != int(cn_profiles.iloc[cn_row, 0]):
            cn_row += 1
        while int(counts_liquid_filtered['pos'].iloc[i]) > cn_profiles.iloc[cn_row, 2]:
            cn_row += 1
        res.append(cn_profiles.iloc[cn_row, 3:cn_profiles.shape[1]-1])  # don't need normal so omit last column
    res = pd.DataFrame(data=np.array(res), dtype=float)
    counts_liquid_filtered.drop(columns=['chr', 'pos'], inplace=True)   
    
    # construct an Lx(2+K) matrix (L=num SNVs post filtering, 2+K=ref & alt counts for liquid biopsy + mutant copies estimates at SNV l for each clone j
    # REQUIRES: rows of all dfs are sorted by event_id AND there is at least one read at each SNV (ref_counts + alt_counts > 0)
    counts = [counts_liquid_filtered]
    for clone in temp_clones:
        clone_filtered = clone[clone['event_id'].isin(intersection)] # filter rows
        copy_est = clone_filtered.apply(lambda row: row.alt_counts / (row.ref_counts + row.alt_counts), axis=1) # compute estimate for num mutant copies 
        counts.append(np.reshape(copy_est.to_numpy(), (-1, 1)))
    counts = pd.DataFrame(data=np.concatenate(counts, axis=1), dtype=float)
    counts.iloc[:, 2:] = counts.iloc[:, 2:].to_numpy() * res.to_numpy()  # multiply estimate for num mutant copies by corresponding CN
    # counts.iloc[:, 2:] = counts.iloc[:, 2:].apply(lambda row: row / row.sum(), axis=1)  # normalize (might not need to if using BinomialLogit in model)
    return counts.to_numpy()
