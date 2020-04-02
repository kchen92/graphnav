from __future__ import division


class SuccessTracker(object):

    def __init__(self):
        self._n_attempts = 0
        self._n_successes = 0

    def update_tracker(self, is_success):
        assert isinstance(is_success, bool)
        self._n_attempts += 1
        if is_success is True:
            self._n_successes += 1

    @property
    def n_attempts(self):
        return self._n_attempts

    @property
    def n_successes(self):
        return self._n_successes

    @property
    def success_rate(self):
        if self.n_attempts == 0:
            return None

        return float(self.n_successes) / self.n_attempts
