from src.inference import run_inference
from src.preprocessing import preprocess_cn_configs, get_reads, preprocess_bam_file
from src.utils import save_results


def run(input_path,
        gc,
        mapp,
        cn_profiles_path,
        output,
        model,
        num_samples,
        num_warmup,
        seed,
        progress_bar,
        chrs,
        bin_size,
        qual,
        verbose):

    raw_data, raw_cn_profiles = preprocess_bam_file(input_path, cn_profiles_path, chrs, bin_size, qual, gc, mapp, verbose)
    data, cn_profiles = preprocess_cn_configs(raw_data, raw_cn_profiles, verbose)

    sampler_obj = run_inference(model,
                                data.squeeze(),
                                cn_profiles.squeeze(),
                                num_samples,
                                num_warmup,
                                int(seed),
                                progress_bar,
                                verbose)

    save_results(output, sampler_obj, cn_profiles.shape[1]-1, verbose)
