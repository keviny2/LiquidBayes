from src.inference import run_inference
from src.preprocessing import preprocess_cn_configs, get_reads, preprocess_bam_file
from src.process_snvs import get_counts, combine_counts
from src.utils import save_results, _print


def run(input_path,
        cn_profiles_path,
        output,
        liquid_vcf,
        tissue_bams,
        tissue_vcfs,
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
        verbose):

    # load data and preprocess
    raw_data, raw_cn_profiles = preprocess_bam_file(input_path, cn_profiles_path, chrs, bin_size, qual, gc, mapp, verbose)
    data, cn_profiles = preprocess_cn_configs(raw_data, raw_cn_profiles, verbose)

    # get counts at SNV locations if applicable
    if tissue_bams == ('',) and tissue_vcfs == ('',):
        counts = None
    else:
        counts_liquid = get_counts(input_path, liquid_vcf, verbose)
        counts_clones = []
        for i in range(len(tissue_vcfs)):
            counts_clones.append(get_counts(tissue_bams[i], tissue_vcfs[i], verbose))
        counts = combine_counts(counts_liquid, counts_clones, verbose)

    sampler_obj = run_inference(model,
                                data.squeeze(),
                                cn_profiles.squeeze(),
                                counts,
                                num_samples,
                                num_warmup,
                                int(seed),
                                progress_bar,
                                verbose)

    if model in ['cn', 'cn_snv']:
        save_results(model, output, sampler_obj, cn_profiles.shape[1]-1, verbose)
    elif model == 'one-more-clone':
        save_results(model, output, sampler_obj, cn_profiles.shape[1], verbose)
    else:
        _print('Invalid model', verbose)
