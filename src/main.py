from src.inference import run_inference
from src.preprocessing import remove_outliers, get_reads, preprocess_bam_file
from src.process_snvs import get_counts, process_counts
from src.utils import save_results, _print


def run(liquid_bam,
        cn_profiles,
        output,
        liquid_vcf,
        clone_bams,
        clone_vcfs,
        model,
        num_samples,
        num_warmup,
        seed,
        gc,
        mapp,
        progress_bar,
        chrs,
        bin_size,
        qual,
        verbose,
        temp_dir):

    # load data and preprocess
    raw_data, raw_cn_profiles = preprocess_bam_file(liquid_bam, cn_profiles, chrs, bin_size, qual, gc, mapp, verbose, temp_dir)
    data, cn_profiles = remove_outliers(raw_data, raw_cn_profiles, verbose)

    # get counts at SNV locations if applicable
    if clone_bams == ('',) and clone_vcfs == ('',) or model == 'cn':
        counts = None
    else:
        counts_liquid = get_counts(liquid_bam, liquid_vcf, verbose)
        counts_clones = []
        for i in range(len(clone_vcfs)):
            counts_clones.append(get_counts(clone_bams[i], clone_vcfs[i], verbose))
        counts = process_counts(counts_liquid, counts_clones, cn_profiles, verbose)

    cn_profiles = cn_profiles[:, 3:].squeeze()  # first three columns are genomic bin information which we don't need for inference
    data = data.squeeze()

    sampler_obj = run_inference(model,
                                data,
                                cn_profiles, 
                                counts,
                                num_samples,
                                num_warmup,
                                int(seed),
                                progress_bar,
                                verbose)

    save_results(output, sampler_obj, cn_profiles.shape[1]-1, verbose)
