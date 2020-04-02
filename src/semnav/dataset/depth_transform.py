"""Transform for Gibson inputs into depth values.
"""

import numpy as np


class DepthTransform(object):
    """Process depth image.
    """

    def __init__(self, min_depth, max_depth, hole_val=0.):
        """Constructor.

        Args:
            min_depth: Minimum distance detectable by the sensor in meters.
            max_depth: Maximum distance detectable by the sensor in meters.
            hole_val: Holes in the depth map are set to this value (float32).
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.hole_val = hole_val

    def __call__(self, depth_sample):
        """Assumes depth_sample is in meters.
        """
        # Replace nans with hole_val
        depth_no_nan = np.where(np.isnan(depth_sample), self.hole_val, depth_sample)
        clipped = np.clip(depth_no_nan, self.min_depth, self.max_depth)  # Clip values
        return clipped
