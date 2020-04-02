"""Sample items in a list, given a probability distribution.
"""

from __future__ import division

import numpy as np


class DiscreteSampler(object):
    """Sample items in a list, given a probability distribution.
    """

    def __init__(self, item_list, unnormalized_prob):
        self.items = item_list
        self.p = self.normalize_prob(unnormalized_prob)
        assert len(item_list) == self.p.shape[0]

    @staticmethod
    def normalize_prob(unnormalized_prob):
        unnormalized_prob = np.array(unnormalized_prob)
        assert unnormalized_prob.ndim == 1
        return unnormalized_prob / np.sum(unnormalized_prob)

    def sample_once(self):
        return np.random.choice(self.items, p=self.p)
