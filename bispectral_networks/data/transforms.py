import numpy as np
import torch
import itertools
from collections import OrderedDict
from skimage.transform import rotate


class Transform:
    def __init__(self):
        self.name = None

    def define_containers(self, tlabels):
        transformed_data, transforms, new_labels = [], [], []
        new_tlabels = OrderedDict({k: [] for k in tlabels.keys()})
        return transformed_data, new_labels, new_tlabels, transforms

    def reformat(self, transformed_data, new_labels, new_tlabels, transforms):
        try:
            transformed_data = torch.stack(transformed_data)
        except:
            transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        # new_labels = torch.tensor(new_labels)
        new_labels = torch.stack(new_labels)
        for k in new_tlabels.keys():
            new_tlabels[k] = torch.stack(new_tlabels[k])
        return transformed_data, new_labels, new_tlabels, transforms


class CenterMean(Transform):
    def __init__(self):
        super().__init__()
        self.name = "center-mean"

    def __call__(self, data, labels, tlabels):
        if len(data.shape) == 2:
            axis = -1
        elif len(data.shape) == 3:
            axis = (-1, -2)
        else:
            raise ValueError(
                "Operation is not defined for data of dimension {}".format(
                    len(data.shape)
                )
            )
        means = data.mean(axis=axis, keepdims=True)
        transformed_data = data - means
        return transformed_data, labels, tlabels, means


class UnitStd(Transform):
    def __init__(self):
        super().__init__()
        self.name = "unit-std"

    def __call__(self, data, labels, tlabels):
        if len(data.shape) == 2:
            axis = -1
        elif len(data.shape) == 3:
            axis = (-1, -2)
        else:
            raise ValueError(
                "Operation is not defined for data of dimension {}".format(
                    len(data.shape)
                )
            )
        stds = data.std(axis=axis, keepdims=True)
        transformed_data = data / stds
        return transformed_data, labels, tlabels, stds

    
class Ravel(Transform):
    def __init__(self):
        super().__init__()
        self.name = "ravel"

    def __call__(self, data, labels, tlabels):
        transformed_data = data.reshape(data.shape[0], -1)
        transforms = torch.zeros(len(data))
        return transformed_data, labels, tlabels, transforms
    
    
class CircleCrop(Transform):
    def __init__(self):
        super().__init__()
        self.name = "circle-crop"

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        img_size = data.shape[1:]

        v, h = np.mgrid[: img_size[0], : img_size[1]]
        equation = (v - ((img_size[0] - 1) / 2)) ** 2 + (
            h - ((img_size[1] - 1) / 2)
        ) ** 2
        circle = equation < (equation.max() / 2)

        transformed_data = data.clone()
        transformed_data[:, ~circle] = 0.0
        transforms = torch.zeros(len(data))

        return transformed_data, labels, tlabels, transforms

    
class CyclicTranslation2D(Transform):

    def __init__(self, fraction_transforms=0.1, sample_method="linspace"):
        super().__init__()
        assert sample_method in [
            "linspace",
            "random",
        ], "sample_method must be one of ['linspace', 'random']"
        self.fraction_transforms = fraction_transforms
        self.sample_method = sample_method
        self.name = "cyclic-translation-2d"

    def get_samples(self, dim_v, dim_h):
        n_transforms = int(self.fraction_transforms * dim_h * dim_v)
        if self.sample_method == "linspace":
            unit_v = dim_v / n_transforms
            unit_h = dim_h / n_transforms
            return [
                (int(v), int(h))
                for v, h in zip(
                    np.arange(0, dim_v, unit_v),
                    np.arange(0, dim_h, unit_h),
                )
            ]
        else:
            all_transforms = list(
                itertools.product(
                    np.arange(dim_v),
                    np.arange(dim_h),
                )
            )
            select_transforms_idx = np.random.choice(
                range(len(all_transforms)), size=n_transforms, replace=False
            )
            select_transforms = [
                all_transforms[x] for x in sorted(select_transforms_idx)
            ]
            return select_transforms

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

        dim_v, dim_h = data.shape[-2:]
        select_transforms = self.get_samples(dim_v, dim_h)
        for i, x in enumerate(data):
            if self.sample_method == "random" and self.fraction_transforms != 1.0:
                select_transforms = self.get_samples(dim_v, dim_h)
            for tv, th in select_transforms:
                xt = torch.roll(x, (tv, th), dims=(-2, -1))
                transformed_data.append(xt)
                transforms.append((int(tv), int(th)))
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms


class SO2(Transform):
    def __init__(self, fraction_transforms=0.1, sample_method="linspace"):
        super().__init__()
        assert sample_method in [
            "linspace",
            "random",
        ], "sample_method must be one of ['linspace', 'random']"
        self.fraction_transforms = fraction_transforms
        self.sample_method = sample_method
        self.name = "so2"

    def get_samples(self):
        n_transforms = int(self.fraction_transforms * 360)
        if self.sample_method == "linspace":
            return np.linspace(0, 359, n_transforms)
        else:
            select_transforms = np.random.choice(
                np.arange(360), size=n_transforms, replace=False
            )
            select_transforms = sorted(select_transforms)
            return select_transforms

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

        select_transforms = self.get_samples()
        for i, x in enumerate(data):
            if self.sample_method == "random":
                select_transforms = self.get_samples()
            for t in select_transforms:
                xt = rotate(x, t)
                transformed_data.append(xt)
                transforms.append(t)
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms