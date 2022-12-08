from typing import List

import numpy as np
from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.spatial.dictionary import RandAffined
from monai.transforms.utils import create_rotate
from monai.transforms.utility.dictionary import ToTensord
from monai.config import KeysCollection


class DeleteChannelsd(MapTransform):
    """Delete specified channels of keyed entries"""

    def __init__(
        self,
        keys: KeysCollection,
        channels: List[int],
        allow_missing_keys: bool = False,
    ) -> None:

        super().__init__(keys, allow_missing_keys)
        self.channels = channels

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            data = d[key]

            data = np.delete(data, self.channels, axis=1)
            d[key] = data

        return d


class ExtractSegmentationLabeld(MapTransform):
    """Extracting specified dimension as a segmentation label for pointcloud"""

    def __init__(
        self,
        pcd_key: KeysCollection,
        label_dim: int = -1,
        label_name: str = "label",
        binarize_threshold: float = None,
        allow_missing_keys: bool = False,
    ) -> None:

        super().__init__(pcd_key, allow_missing_keys)
        self.pcd_key = pcd_key
        self.label_dim = label_dim
        self.label_name = label_name
        self.binarize_threshold = binarize_threshold

    def __call__(self, data):
        d = dict(data)
        pcd = d[self.pcd_key]

        input, label = pcd[:, : self.label_dim], pcd[:, self.label_dim]

        if self.binarize_threshold is not None:
            label = (label > self.binarize_threshold).astype(float)

        d[self.pcd_key] = input
        d[self.label_name] = label

        return d

class ToFloatTensord(ToTensord):
    """Cast numpy to float32 tensor"""

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key]).float()
        return d


class PointcloudRandomSubsampled(Randomizable, MapTransform):
    """Subsampling pointcloud to specified number of points"""

    def __init__(
        self,
        keys: KeysCollection,
        sub_size: int,
        shuffle: bool = True,
        allow_missing_keys: bool = False,
    ):

        MapTransform.__init__(self, keys, allow_missing_keys)
        self.sub_size = sub_size
        self.shuffle = shuffle

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            data = d[key]

            permutation = self.R.permutation(len(data))
            indices = permutation[: self.sub_size]

            data = data[indices]
            d[key] = data

        return d


class PointcloudRandomAffined(RandAffined):
    """Random rotation applied to pointcloud coordinates"""

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        rotate_range=None,
        translate_range=None,
        allow_missing_keys: bool = False,
    ):

        super().__init__(
            keys=keys,
            prob=prob,
            rotate_range=rotate_range,
            translate_range=translate_range,
            allow_missing_keys=allow_missing_keys,
        )

    def __call__(self, data):
        d = dict(data)
        self.randomize()

        for key in self.key_iterator(d):
            affine = self.rand_affine
            rotation = create_rotate(3, affine.rand_affine_grid.rotate_params)

            coords = d[key][:, :3]
            coords = np.c_[coords, np.ones(len(coords))]

            rotated_coords = (rotation @ coords.T).T
            rotated_coords = rotated_coords[:, :3]

            d[key][:, :3] = rotated_coords

        return d


class PointcloudRandomJitterd(Randomizable, MapTransform):
    """Random jitter applied to pointcloud coordinates"""

    def __init__(
        self, keys: KeysCollection, std: float = 0.01, allow_missing_keys: bool = False,
    ):

        MapTransform.__init__(self, keys, allow_missing_keys)
        self.std = std

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            data = d[key]
            jitter = self.R.normal(0.0, self.std, size=(data.shape[0], 3))

            data[:, :3] += jitter
            d[key] = data

        return d
