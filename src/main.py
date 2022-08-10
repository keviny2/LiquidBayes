import numpy as np
from src.inference import run_inference
from src.preprocessing import preprocess_cn_configs, get_reads, preprocess_bam_file
from src.utils import load_data, save_results, blockPrint


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
        preprocess,
        chrs,
        bin_size,
        qual,
        verbose):
    if not verbose:
        blockPrint()

    if input_path.endswith('.bam'):
        raw_data, raw_cn_profiles = preprocess_bam_file(input_path, cn_profiles_path, chrs, bin_size, qual, gc, mapp)
    elif input_path.endswith('.bed'):
        raw_data, raw_cn_profiles = load_data(input_path, cn_profiles_path)

    if preprocess:
        data, cn_profiles = preprocess_cn_configs(raw_data, raw_cn_profiles)
    else:
        data, cn_profiles = raw_data, raw_cn_profiles
    sampler_obj = run_inference(model,
                                data.squeeze(),
                                cn_profiles.squeeze(),
                                num_samples,
                                num_warmup,
                                int(seed),
                                progress_bar)
    if model == 'simple':
        save_results(model, output, sampler_obj, cn_profiles.shape[1]-1)
    else:
        save_results(model, output, sampler_obj, cn_profiles.shape[1])
