import os
from os.path import basename, splitext
from functools import partial
from glob import glob
from hashlib import sha1
from typing import Callable, Optional, Tuple

from glog import logger

from model_training.common import dataset
from model_training.common.aug import get_transforms


def hash_from_paths(paths: Tuple[str, str], salt: str = "") -> str:
    path_a, path_b, _ = paths
    names = "".join(map(os.path.basename, (path_a, path_b)))
    return sha1(f"{names}_{salt}".encode()).hexdigest()


class PairedDataset(dataset.AbstractDataset):
    def __init__(
        self,
        files_a: Tuple[str],
        files_b: Tuple[str],
        names: Tuple[str],
        transform: Callable,
        preload: bool = True,
        preload_size: Optional[int] = 0,
        verbose=True,
    ):

        assert len(files_a) == len(files_b)

        self.names = names
        self.preload = False
        self.data_a = files_a
        self.data_b = files_b
        self.transform = transform
        self.verbose = verbose

        if preload:
            preload_fn = partial(self._bulk_preload, preload_size=preload_size)
            self.data_a, self.data_b = map(preload_fn, (self.data_a, self.data_b))
            self.preload = True

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, idx):
        a, b, name = self.data_a[idx], self.data_b[idx], self.names[idx]
        if not self.preload:
            a, b = map(dataset.read_img, (a, b))
        a, b = map(self._preprocess, self.transform(a, b))
        return {"img": a, "mask": b, "name": name}

    @staticmethod
    def from_config(config):
        files_a = sorted(glob(config["files_a"], recursive=True))
        files_b = sorted(glob(config["files_b"], recursive=True))

        logger.info(
            "files_a read: {} files., files_b read: {} files.".format(
                len(files_a), len(files_b)
            )
        )

        names = list(map(lambda path: splitext(basename(path))[0], files_a))

        transform = get_transforms(config["transform"])

        # ToDo: make augmentations more customizible via transform

        hash_fn = hash_from_paths
        # ToDo: add more hash functions
        verbose = config.get("verbose", True)
        data = dataset.subsample(
            data=zip(files_a, files_b, names),
            bounds=config.get("bounds", (0, 1)),
            hash_fn=hash_fn,
            verbose=verbose,
        )

        files_a, files_b, names = map(list, zip(*data))

        return PairedDataset(
            files_a=files_a,
            files_b=files_b,
            names=names,
            preload=config["preload"],
            preload_size=config["preload_size"],
            transform=transform,
            verbose=verbose,
        )
