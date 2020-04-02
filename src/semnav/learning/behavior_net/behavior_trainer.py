from __future__ import print_function
from __future__ import division

import torch

from semnav.learning.trainer import Trainer
from semnav.learning.behavior_net.behavior_evaluator import BehaviorEvaluator


class BehaviorTrainer(Trainer):

    def __init__(self):
        super(BehaviorTrainer, self).__init__()

    @staticmethod
    def get_decode_type(decode_type):
        return BehaviorEvaluator.get_decode_type(decode_type)

    def get_criterion(self):
        return torch.nn.MSELoss()

    def get_evaluator(self):
        return BehaviorEvaluator()


if __name__ == '__main__':
    BehaviorTrainer().train()
