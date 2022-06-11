import os
import pyro.distributions as dist
import torch

# https://forum.pyro.ai/t/trying-to-write-a-uniform-discrete-distribution/402/6
class UniformDiscrete(torch.distributions.distribution.Distribution):
    def __init__(self, vals, probs):
        self.vals = vals
        self.categorical = dist.Categorical(probs)
        super(UniformDiscrete, self).__init__(self.categorical.batch_shape,
                                              self.categorical.event_shape)
    def sample(self, sample_shape=torch.Size()):
        return self.vals[self.categorical.sample(sample_shape)]
    def log_prob(self, value):
        idx = (self.vals == value).nonzero()
        return self.categorical.log_prob(idx)
