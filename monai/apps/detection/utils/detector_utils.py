# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from monai.data.box_utils import standardize_empty_box
from monai.transforms.croppad.array import SpatialPad
from monai.transforms.utils import compute_divisible_spatial_size, convert_pad_mode
from monai.utils import PytorchPadMode, ensure_tuple_rep


def check_input_images(input_images: list[Tensor] | Tensor, spatial_dims: int) -> None:
    """
    Validate the input dimensionality (raise a `ValueError` if invalid).

    Args:
        input_images: It can be 1) a tensor sized (B, C, H, W) or  (B, C, H, W, D),
            or 2) a list of image tensors, each image i may have different size (C, H_i, W_i) or  (C, H_i, W_i, D_i).
        spatial_dims: number of spatial dimensions of the images, 2 or 3.
    """
    if isinstance(input_images, Tensor):
        if len(input_images.shape) != spatial_dims + 2:
            raise ValueError(
                "When input_images is a Tensor, its need to be (spatial_dims + 2)-D."
                f"In this case, it should be a {(spatial_dims + 2)}-D Tensor, got Tensor shape {input_images.shape}."
            )
        if not torch.isfinite(input_images).all():
            raise ValueError("input image is NaN or Inf.")
    elif isinstance(input_images, list):
        for idx, img in enumerate(input_images):
            if len(img.shape) != spatial_dims + 1:
                raise ValueError(
                    "When input_images is a List[Tensor], each element should have be (spatial_dims + 1)-D."
                    f"In this case, it should be a {(spatial_dims + 1)}-D Tensor, got Tensor shape {img.shape}."
                )
            if not torch.isfinite(img).all():
                raise ValueError(f"input image at index {idx} is NaN or Inf.")
    else:
        raise ValueError("input_images needs to be a List[Tensor] or Tensor.")
    return


def check_training_targets(
    input_images: list[Tensor] | Tensor,
    targets: list[dict[str, Tensor]] | None,
    spatial_dims: int,
    target_label_key: str,
    target_box_key: str,
    num_classes: int | None = None,
) -> list[dict[str, Tensor]]:
    """
    Validate the input images/targets during training (raise a `ValueError` if invalid).

    Args:
        input_images: It can be 1) a tensor sized (B, C, H, W) or  (B, C, H, W, D),
            or 2) a list of image tensors, each image i may have different size (C, H_i, W_i) or  (C, H_i, W_i, D_i).
        targets: a list of dict. Each dict with two keys: target_box_key and target_label_key,
            ground-truth boxes present in the image.
        spatial_dims: number of spatial dimensions of the images, 2 or 3.
        target_label_key: the expected key of target labels.
        target_box_key: the expected key of target boxes.
    """
    if targets is None:
        raise ValueError("Please provide ground truth targets during training.")

    if len(input_images) != len(targets):
        raise ValueError(f"len(input_images) should equal to len(targets), got {len(input_images)}, {len(targets)}.")

    for i in range(len(targets)):
        target = targets[i]
        if (target_label_key not in target.keys()) or (target_box_key not in target.keys()):
            raise ValueError(
                f"{target_label_key} and {target_box_key} are expected keys in targets. Got {target.keys()}."
            )

        boxes = target[target_box_key]
        if not isinstance(boxes, torch.Tensor):
            raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
        if len(boxes.shape) != 2 or boxes.shape[-1] != 2 * spatial_dims:
            if boxes.numel() == 0:
                warnings.warn(
                    f"Warning: Given target boxes has shape of {boxes.shape}. "
                    f"The detector reshaped it with boxes = torch.reshape(boxes, [0, {2 * spatial_dims}])."
                )
            else:
                raise ValueError(
                    f"Expected target boxes to be a tensor of shape [N, {2 * spatial_dims}], got {boxes.shape}.)."
                )
        if not torch.is_floating_point(boxes):
            raise ValueError(f"Expected target boxes to be a float tensor, got {boxes.dtype}.")
        boxes = standardize_empty_box(boxes, spatial_dims=spatial_dims)  # type: ignore[assignment]
        if not torch.isfinite(boxes).all():
            raise ValueError("target boxes are NaN or Inf.")
        if boxes.numel() > 0:
            box_sizes = boxes[:, spatial_dims:] - boxes[:, :spatial_dims]
            if (box_sizes <= 0).any():
                raise ValueError("target boxes must have positive size in every spatial dimension.")
        targets[i][target_box_key] = boxes

        labels = target[target_label_key]
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"Expected target labels to be of type Tensor, got {type(labels)}.")
        if torch.is_floating_point(labels):
            if not torch.isfinite(labels).all():
                raise ValueError("target labels are NaN or Inf.")
            warnings.warn(f"Warning: Given target labels is {labels.dtype}. The detector converted it to torch.long.")
            labels = labels.long()
        if labels.ndim != 1:
            raise ValueError(f"Expected target labels to be a 1D tensor, got shape {labels.shape}.")
        if labels.shape[0] != boxes.shape[0]:
            raise ValueError("target labels and target boxes must contain the same number of entries.")
        if num_classes is not None and labels.numel() > 0:
            if labels.min() < 0 or labels.max() >= num_classes:
                raise ValueError(f"target labels must be in range [0, {num_classes}).")
        targets[i][target_label_key] = labels
    return targets


def pad_images(
    input_images: list[Tensor] | Tensor,
    spatial_dims: int,
    size_divisible: int | Sequence[int],
    mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    **kwargs: Any,
) -> tuple[Tensor, list[list[int]]]:
    """
    Pad the input images, so that the output spatial sizes are divisible by `size_divisible`.
    It pads them at the end to create a (B, C, H, W) or (B, C, H, W, D) Tensor.
    Padded size (H, W) or (H, W, D) is divisible by size_divisible.
    Default padding uses constant padding with value 0.0

    Args:
        input_images: It can be 1) a tensor sized (B, C, H, W) or  (B, C, H, W, D),
            or 2) a list of image tensors, each image i may have different size (C, H_i, W_i) or  (C, H_i, W_i, D_i).
        spatial_dims: number of spatial dimensions of the images, 2D or 3D.
        size_divisible: int or Sequence[int], is the expected pattern on the input image shape.
            If an int, the same `size_divisible` will be applied to all the input spatial dimensions.
        mode: available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for `torch.pad` function.

    Return:
        - images, a (B, C, H, W) or (B, C, H, W, D) Tensor
        - image_sizes, the original spatial size of each image
    """
    size_divisible = ensure_tuple_rep(size_divisible, spatial_dims)

    # If input_images: Tensor
    if isinstance(input_images, Tensor):
        orig_size = list(input_images.shape[-spatial_dims:])
        new_size = compute_divisible_spatial_size(spatial_shape=orig_size, k=size_divisible)
        all_pad_width = [(0, max(sp_i - orig_size[i], 0)) for i, sp_i in enumerate(new_size)]
        pt_pad_width = [val for sublist in all_pad_width for val in sublist[::-1]][::-1]
        if max(pt_pad_width) == 0:
            # if there is no need to pad
            return input_images, [orig_size] * input_images.shape[0]
        mode_: str = convert_pad_mode(dst=input_images, mode=mode)
        return F.pad(input_images, pt_pad_width, mode=mode_, **kwargs), [orig_size] * input_images.shape[0]

    # If input_images: List[Tensor])
    image_sizes = [img.shape[-spatial_dims:] for img in input_images]
    in_channels = input_images[0].shape[0]
    dtype = input_images[0].dtype
    device = input_images[0].device

    # compute max_spatial_size
    image_sizes_t = torch.tensor(image_sizes)
    max_spatial_size_t, _ = torch.max(image_sizes_t, dim=0)

    if len(max_spatial_size_t) != spatial_dims or len(size_divisible) != spatial_dims:
        raise ValueError(" Require len(max_spatial_size_t) == spatial_dims ==len(size_divisible).")

    max_spatial_size = compute_divisible_spatial_size(spatial_shape=list(max_spatial_size_t), k=size_divisible)

    # allocate memory for the padded images
    images = torch.zeros([len(image_sizes), in_channels] + list(max_spatial_size), dtype=dtype, device=device)

    # Use `SpatialPad` to match sizes, padding in the end will not affect boxes
    padder = SpatialPad(spatial_size=max_spatial_size, method="end", mode=mode, **kwargs)
    for idx, img in enumerate(input_images):
        images[idx, ...] = padder(img)

    return images, [list(ss) for ss in image_sizes]


def preprocess_images(
    input_images: list[Tensor] | Tensor,
    spatial_dims: int,
    size_divisible: int | Sequence[int],
    mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    **kwargs: Any,
) -> tuple[Tensor, list[list[int]]]:
    """
    Preprocess the input images, including

    - validate of the inputs
    - pad the inputs so that the output spatial sizes are divisible by `size_divisible`.
      It pads them at the end to create a (B, C, H, W) or (B, C, H, W, D) Tensor.
      Padded size (H, W) or (H, W, D) is divisible by size_divisible.
      Default padding uses constant padding with value 0.0

    Args:
        input_images: It can be 1) a tensor sized (B, C, H, W) or  (B, C, H, W, D),
            or 2) a list of image tensors, each image i may have different size (C, H_i, W_i) or  (C, H_i, W_i, D_i).
        spatial_dims: number of spatial dimensions of the images, 2 or 3.
        size_divisible: int or Sequence[int], is the expected pattern on the input image shape.
            If an int, the same `size_divisible` will be applied to all the input spatial dimensions.
        mode: available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for `torch.pad` function.

    Return:
        - images, a (B, C, H, W) or (B, C, H, W, D) Tensor
        - image_sizes, the original spatial size of each image
    """
    check_input_images(input_images, spatial_dims)
    size_divisible = ensure_tuple_rep(size_divisible, spatial_dims)

    return pad_images(input_images, spatial_dims, size_divisible, mode, **kwargs)
