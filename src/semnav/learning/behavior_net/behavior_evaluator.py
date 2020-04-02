from __future__ import print_function
from __future__ import division

from semnav.learning.evaluator import Evaluator


class BehaviorEvaluator(Evaluator):

    def __init__(self, cfg=None):
        super(BehaviorEvaluator, self).__init__(cfg=cfg)

    @staticmethod
    def get_decode_type(decode_type):
        if decode_type == 'temporal':
            return 'temporal'
        elif decode_type == 'single_frame':
            return 'single_frame'
        else:
            raise ValueError


if __name__ == '__main__':
    BehaviorEvaluator().test()
