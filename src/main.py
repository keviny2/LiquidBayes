from src.inference import run_inference
from src.preprocessing import preprocess_cn_configs, get_reads, preprocess_bam_file
from src.utils import load_data, save_results, blockPrint


def run(input_path,
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
        print('Processing .bam file')
        raw_data, raw_cn_profiles = preprocess_bam_file(input_path, cn_profiles_path, chrs, bin_size, qual)
    elif input_path.endswith('.bed'):
        print('Processing .bed file')
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

    save_results(output, sampler_obj, cn_profiles.shape[1]-1)
