from functools import partial
from typing import Callable, Iterable, Tuple

import cv2
import numpy as np

from joblib import Parallel, cpu_count, delayed
from glog import logger

from skimage.io import imread
from torch.utils import data
from tqdm import tqdm


def read_img(x: str):
    img = cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
    if img is None:
        logger.warning(f"Can not read image {x} with OpenCV, switching to scikit-image")
        img = imread(x)
    return img


def subsample(
    data: Iterable,
    bounds: Tuple[float, float],
    hash_fn: Callable,
    n_buckets=100,
    salt="",
    verbose=True,
):
    data = list(data)
    buckets = split_into_buckets(data, n_buckets=n_buckets, salt=salt, hash_fn=hash_fn)

    lower_bound, upper_bound = [x * n_buckets for x in bounds]
    msg = (
        f"Subsampling buckets from {lower_bound} to {upper_bound}, "
        f"total buckets number is {n_buckets}"
    )
    if salt:
        msg += f"; salt is {salt}"
    if verbose:
        logger.info(msg)
    return np.array(
        [
            sample
            for bucket, sample in zip(buckets, data)
            if lower_bound <= bucket < upper_bound
        ]
    )


def split_into_buckets(data: Iterable, n_buckets: int, hash_fn: Callable, salt=""):
    hashes = map(partial(hash_fn, salt=salt), data)
    return np.array([int(x, 16) % n_buckets for x in hashes])


class AbstractDataset(data.Dataset):
    @staticmethod
    def _preprocess(img):
        img = np.transpose(img, (2, 0, 1))
        return img

    def _bulk_preload(self, data: Iterable[str], preload_size: int):
        jobs = [delayed(self._preload)(x, preload_size=preload_size) for x in data]
        jobs = tqdm(jobs, desc="preloading images", disable=not self.verbose)
        return Parallel(n_jobs=cpu_count(), backend="threading")(jobs)

    @staticmethod
    def _preload(x: str, preload_size: int):
        img = read_img(x)
        if preload_size:
            h, w, *_ = img.shape
            h_scale = preload_size / h
            w_scale = preload_size / w
            scale = max(h_scale, w_scale)
            img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
            assert min(img.shape[:2]) >= preload_size, f"weird img shape: {img.shape}"
        return img

    @staticmethod
    def from_config(config):
        raise NotImplementedError()
