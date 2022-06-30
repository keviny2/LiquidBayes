import pandas as pd

def pyro_mcmc(sampler_obj):
    samples = pd.DataFrame.from_dict(sampler_obj.get_samples())  # get samples from inference
    num_subclones = cn_profiles.shape[1]-1  # cn_profiles includes normal CN profile, so minus 1
    clones = list(string.ascii_uppercase)[:num_subclones] + ['normal']
    rhos = pd.DataFrame(samples['rho'].to_list(), columns=clones, dtype=float) 
    samples.join(rhos).drop('rho', axis=1).describe().loc[['mean']].to_csv(output)  # write mean of each sample site to csv file

