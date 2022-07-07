import numpy as np

from src.inference import run_inference
from src.utils import load_data, preprocess_data, save_results


def run(input_path,
        cn_profiles_path,
        output,
        model,
        num_samples,
        num_warmup,
        seed,
        progress_bar,
        preprocess):

    raw_data, raw_cn_profiles = load_data(input_path, cn_profiles_path)

    if preprocess:
        data, cn_profiles = preprocess_data(raw_data, raw_cn_profiles)
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
