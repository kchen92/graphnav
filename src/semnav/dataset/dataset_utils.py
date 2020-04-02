import os

from semnav.lib.utils import get_frame_paths, get_frame_idx


def prepend_dataset_root(cfg, path):
    assert not path.startswith(cfg.dataset_root)
    return os.path.join(cfg.dataset_root, path)


def remove_dataset_root(cfg, path):
    assert path.startswith(cfg.dataset_root)
    return os.path.relpath(path, start=cfg.dataset_root)
