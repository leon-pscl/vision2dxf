"""BMnet-style CNN for 14-measurement estimation from front/side silhouettes.

Architecture
------------
``[front | side]`` concatenated spatially to ``(img_height, 2 * img_width)`` and carried in a
**single mask channel**, fed to an ImageNet-pretrained MNASNet trunk, average-pooled, and
concatenated with normalised height (and, for V-HW, weight) scalars immediately before the
MLP head.

Channel layout
--------------
The assembled tensor is ``[mask, height, weight]`` for V-HW (3 channels) and
``[mask, height]`` for V-H (2 channels). The constant scalar channels are still *emitted* by
:func:`to_tensor` so the tensor contract is a single, stable, testable object - but the
network does **not** consume them as spatial input. It reads their scalar value out at the
head. The trailing constant ``ones`` channel that the previous version appended has been
removed: it carried no information and only existed to make the concatenation unambiguous,
which the channel count already does.

Stem adaptation
---------------
The ImageNet stem expects 3 input channels and the mask is 1. The pretrained weights are
**mean-reduced over the input-channel axis** to ``(out, 1, k, k)``, which is the standard
grayscale adaptation: it preserves every pretrained output filter with full spatial structure
instead of discarding the first layer. See DECISIONS.md D5.

Scalar injection (fix B1)
-------------------------
The previous version summed a zero-initialised ``1x1`` projection of the scalar channels at
the second convolution. That path was **dead**: ``scalar_proj`` and ``scalar_fuse`` were both
zero-initialised and applied in series, so ``d(out)/d(scalar_proj.weight)`` is proportional to
``scalar_fuse.weight``, which is zero, and ``d(out)/d(scalar_fuse.weight)`` is proportional to
``scalar_proj(scalars)``, which is zero. Only ``scalar_fuse.bias`` received gradient - a
constant offset that cannot see the scalars. Both variants were therefore silhouette-only
regardless of the headline. It was also shape-incompatible: a 1280-channel full-resolution
tensor added to a 32-channel half-resolution stem output.

Scalars now enter at the head, where a single linear layer can weight them immediately. See
DECISIONS.md D15.

References
----------
Ruiz, N., Bellver, M., Bolkart, T., Arora, A., Lin, M. C., Romero, J., & Bala, R. (2022).
Human body measurement estimation with adversarial augmentation (arXiv:2210.05667).
https://arxiv.org/abs/2210.05667
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

#: Channels of the assembled input tensor: [mask, height, weight] for V-HW (3) and
#: [mask, height] for V-H (2). The trailing constant `ones` channel has been removed: it carried
#: no information and only made the channel count ambiguous. Note the previous version declared
#: N_CHANNELS = 5 while :func:`to_tensor` actually built 4, so every channel-count assertion in
#: the notebook disagreed with the tensor it was checking.
N_CHANNELS = 3
N_CHANNELS_VH = 2

#: Normalisation constants for the constant-valued scalar channels.
HEIGHT_SCALE = 200.0   # height_cm / HEIGHT_SCALE
WEIGHT_SCALE = 100.0   # weight_kg / WEIGHT_SCALE

#: torchvision MNASNet names, printed when the configured backbone is unknown (fix A2).
MNASNET_NAMES = ("mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3")


def _trunk(net: nn.Module) -> Tuple[nn.Module, nn.Sequential, nn.Module]:
    """Return ``(trunk, first_block, container)`` for a torchvision backbone.

    torchvision's backbone families expose the feature extractor under different names, and the
    previous version assumed ``net.features`` unconditionally. That attribute exists on ResNet,
    EfficientNet, ConvNeXt, SqueezeNet and others, but **not on MNASNet**, whose trunk is called
    ``layers`` - so with the corrected backbone name the network would have raised
    ``AttributeError: 'MNASNet' object has no attribute 'features'`` before training started.
    MNASNet's first entry in ``layers`` is a bare ``nn.Conv2d`` (no norm), so it can be replaced
    directly; other families wrap the stem, and those wrappers are handled by replacing the first
    ``nn.Conv2d`` found inside ``features[0]``.

    Parameters
    ----------
    net : nn.Module
        A torchvision model.

    Returns
    -------
    (trunk, first_block, container) : tuple
        ``trunk`` is the callable feature extractor, ``first_block`` is its first child, and
        ``container`` is the object whose child should be replaced when adapting the stem.

    Raises
    ------
    RuntimeError
        If neither ``features`` nor ``layers`` is present.
    """
    for attr in ("features", "layers"):
        trunk = getattr(net, attr, None)
        if isinstance(trunk, nn.Sequential) and len(trunk) > 0:
            return trunk, trunk[0], trunk
    raise RuntimeError(
        f"cannot locate a feature trunk on {type(net).__name__}; expected a Sequential "
        "attribute named 'features' or 'layers'")


def _first_conv(module: nn.Module):
    """Return ``(parent, index, conv)`` for the first ``nn.Conv2d`` inside ``module``.

    Parameters
    ----------
    module : nn.Module
        Module to search (a stem block).

    Returns
    -------
    tuple or None
        ``(parent_module, attribute_name, conv)``, or None when the module holds no Conv2d.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            return module, name, child
        if isinstance(child, nn.Sequential):
            return module, name, child   # descend one level; the caller re-wraps if needed
    return None


def _feature_dim(net: nn.Module) -> int:
    """Return the backbone's feature width immediately before its classifier.

    Parameters
    ----------
    net : nn.Module
        A torchvision model with a ``features`` trunk and a ``classifier`` head.

    Returns
    -------
    int
        Input width of the first ``nn.Linear`` found in ``classifier``, falling back to 1280 when
        the head has no linear layer.
    """
    for m in getattr(net, "classifier", nn.Sequential()):
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    return 1280


def build_backbone(name: str = "mnasnet1_0", pretrained: bool = True,
                   offline_weights: Optional[str] = None) -> Tuple[nn.Module, str]:
    """Instantiate an ImageNet-pretrained torchvision backbone.

    Parameters
    ----------
    name : str, default 'mnasnet1_0'
        Any torchvision model name. MNASNet is the default because Ruiz et al. (2022) use an
        MNASNet trunk and the deployment budget on Kaggle GPUs is comfortable with it. The
        previous default ``mnasnet1_0_3_0`` is not a torchvision model (fix A2).
    pretrained : bool, default True
        Request ImageNet weights.
    offline_weights : str, optional
        Path to a local ``.pth`` state dict, used when the session has no internet. Checked
        before the download is attempted.

    Returns
    -------
    (module, source) : tuple
        The backbone and a short string describing where the weights came from:
        ``'offline'``, ``'imagenet-download'`` or ``'random'``.

    Raises
    ------
    RuntimeError
        If ``name`` is not a torchvision model. The message lists the available MNASNet names
        when ``name`` looks like an MNASNet variant, because that is the failure the fix
        addresses.
    """
    import torchvision.models as tvm

    if not hasattr(tvm, name):
        extra = ""
        if "mnas" in name.lower():
            extra = (f" Did you mean one of {list(MNASNET_NAMES)}? "
                     f"All torchvision MNASNet names: "
                     f"{[n for n in dir(tvm) if n.startswith('mnasnet') and '_' in n[8:]]}")
        raise RuntimeError(f"unknown backbone {name!r}.{extra}")

    if pretrained and offline_weights and Path(offline_weights).is_file():
        model = getattr(tvm, name)(weights=None)
        state = torch.load(offline_weights, map_location="cpu")
        model.load_state_dict(state)
        return model, "offline"

    if pretrained:
        try:
            weights_enum = getattr(tvm, "get_model_weights")(name).DEFAULT
            return getattr(tvm, name)(weights=weights_enum), "imagenet-download"
        except Exception as exc:
            print(f"[backbone] ImageNet weights unavailable ({exc}); falling back to random init. "
                  "Attach a .pth and set config.yaml:offline_weights for a reproducible run.")
            return getattr(tvm, name)(weights=None), "random"

    return getattr(tvm, name)(weights=None), "random"


class MeasurementNet(nn.Module):
    """Front+side silhouette CNN with a 14-measurement regression head.

    Input channel layout is ``[mask, height, weight]`` - **one** mask channel holding the front
    and side silhouettes concatenated horizontally, then the constant scalar channels. That is
    3 channels for V-HW and 2 for V-H.

    First-convolution adaptation
    ---------------------------
    The ImageNet stem expects 3 input channels, but the mask is a single channel. The stem's
    weights are **mean-reduced over the input-channel axis** to shape ``(out, 1, k, k)``: the
    standard grayscale adaptation, which preserves every pretrained output filter with full
    spatial structure instead of throwing the first layer away.

    Scalars
    -------
    Height and weight are read out of their constant channels and concatenated to the
    average-pooled image features immediately before the MLP head. Nothing is zero-initialised:
    the head's first ``nn.Linear`` sees the scalars directly, so it receives gradient from
    step 1. This is what makes the V-HW vs V-H ablation in section 6 interpretable, and it is
    checked twice in code - once after the first optimiser step, once after training.

    References
    ----------
    Ruiz, N., et al. (2022). Human body measurement estimation with adversarial augmentation.
    """

    def __init__(self, backbone: str = "mnasnet1_0", pretrained: bool = True,
                 n_outputs: int = 14, hidden: int = 512, dropout: float = 0.2,
                 use_weight: bool = True, offline_weights: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        backbone : str, default 'mnasnet1_0'
            torchvision model name for the trunk.
        pretrained : bool, default True
            Request ImageNet weights for the trunk.
        n_outputs : int, default 14
            Number of regression outputs.
        hidden : int, default 512
            Width of the MLP hidden layer.
        dropout : float, default 0.2
            Dropout probability in the head.
        use_weight : bool, default True
            V-HW when True, V-H when False. Also fixes the expected input channel count.
        offline_weights : str, optional
            Local ``.pth`` for the ImageNet trunk.
        """
        super().__init__()
        self.use_weight = use_weight
        self.n_outputs = n_outputs
        self.backbone_name = backbone
        self.hidden = hidden
        self.dropout = dropout
        net, self.weights_source = build_backbone(backbone, pretrained, offline_weights)

        # MNASNet exposes its trunk as `layers`, ResNet and friends as `features`
        trunk, first_block, container = _trunk(net)
        self.trunk_name = type(net).__name__
        self.features = trunk

        # --- adapt the pretrained 3-channel stem to a single mask channel -------------------
        # MNASNet's trunk starts with a bare Conv2d, so it is replaced directly. Other families
        # wrap the stem in a block whose first child is the conv; in that case the conv inside the
        # block is replaced instead, which is the same mean-reduced grayscale adaptation.
        if isinstance(first_block, nn.Conv2d):
            self._replace_stem(container, 0, first_block)
        else:
            found = _first_conv(first_block)
            if found is None:
                raise RuntimeError(
                    f"{self.trunk_name}: could not find the stem Conv2d inside "
                    f"{type(first_block).__name__}")
            block, attr, conv = found
            self._replace_stem(block, attr, conv)

        self.final_pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = _feature_dim(net)
        self.feat_dim = feat_dim
        self.n_scalars = 2 if use_weight else 1

        # Scalars enter here, concatenated to the pooled image features (fix B1).
        self.head = nn.Sequential(
            nn.Linear(feat_dim + self.n_scalars, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_outputs),
        )

    @property
    def n_channels(self) -> int:
        """Expected input channel count for this variant.

        Returns
        -------
        int
            3 for V-HW, 2 for V-H.
        """
        return N_CHANNELS if self.use_weight else N_CHANNELS_VH

    def _replace_stem(self, parent: nn.Module, key, stem_conv: nn.Conv2d) -> None:
        """Swap a 3-input-channel stem convolution for a mean-reduced 1-channel one.

        Parameters
        ----------
        parent : nn.Module
            Module holding the convolution (the trunk, or the block that wraps it).
        key : int or str
            Attribute name or index of the convolution within ``parent``.
        stem_conv : nn.Conv2d
            The original pretrained stem convolution.
        """
        if stem_conv.in_channels == 1:
            self.stem_adaptation = "already single-channel"
            return
        new_conv = nn.Conv2d(1, stem_conv.out_channels, kernel_size=stem_conv.kernel_size,
                             stride=stem_conv.stride, padding=stem_conv.padding,
                             bias=stem_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight.copy_(stem_conv.weight.mean(dim=1, keepdim=True))
            if stem_conv.bias is not None:
                new_conv.bias.copy_(stem_conv.bias)
        # nn.Sequential rejects setattr with an int key; it is indexable instead
        if isinstance(key, int) and isinstance(parent, nn.Sequential):
            parent[key] = new_conv
        else:
            setattr(parent, key, new_conv)
        self.stem_adaptation = (f"mean-reduced {stem_conv.in_channels}->1 input channels "
                               f"({self.trunk_name} trunk, stem "
                               f"{'index ' + str(key) if isinstance(key, int) else key})")

    def stem_in_channels(self) -> int:
        """Input channel count of the adapted stem, for the notebook's assertion.

        Returns
        -------
        int
            1 for the mean-reduced grayscale stem.
        """
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                return int(m.in_channels)
        return -1

    def scalar_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Read the constant scalar channels back out of the input tensor.

        Parameters
        ----------
        x : Tensor, shape (B, C, H, 2W)
            Assembled tensor. Channels 1.. are constant across space by construction.

        Returns
        -------
        Tensor, shape (B, n_scalars)
            ``[height_cm / 200]`` and, for V-HW, ``[weight_kg / 100]``, as columns.
        """
        h = x[:, 1].mean(dim=(1, 2)).unsqueeze(1)          # (B, 1)
        if not self.use_weight:
            return h
        w = x[:, 2].mean(dim=(1, 2)).unsqueeze(1)          # (B, 1)
        return torch.cat([h, w], dim=1)                     # (B, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict measurements.

        Parameters
        ----------
        x : Tensor, shape (B, C, img_height, 2 * img_width)
            Channel order ``[mask, height, weight]`` for V-HW and ``[mask, height]`` for V-H.
            ``mask`` holds the front and side silhouettes concatenated horizontally, each
            resized to ``img_height x img_width`` (fix B2: H > W, portrait, matching the
            native mask aspect).

        Returns
        -------
        Tensor, shape (B, n_outputs)
            Predictions, cm.

        Raises
        ------
        ValueError
            If the channel count does not match ``use_weight``.
        """
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} input channels for "
                             f"use_weight={self.use_weight}, got {x.shape[1]}")

        scalars = self.scalar_channels(x)                   # (B, n_scalars)
        h = self.features(x[:, :1])                         # (B, 32, H/2, W) - mask only
        feat = self.final_pool(h).flatten(1)                # (B, feat_dim)
        z = torch.cat([feat, scalars], dim=1)               # (B, feat_dim + n_scalars)
        return self.head(z)                                 # (B, n_outputs)


def to_tensor(front: np.ndarray, side: np.ndarray, height_cm: float, weight_kg: Optional[float],
              img_height: int = 640, img_width: int = 480, threshold: int = 127,
              use_weight: bool = True) -> np.ndarray:
    """Assemble the CNN input tensor for one silhouette pair. **The canonical preprocessing.**

    This function is the single definition of preprocessing. ``preprocess.py`` in the export
    bundle re-exports it verbatim, so a unit test against saved reference tensors is a real
    test of the deployed pipeline rather than of a lookalike.

    Parameters
    ----------
    front, side : ndarray, shape (H, W)
        Silhouette masks (uint8 0/255 or bool).
    height_cm : float
        Subject stature, cm.
    weight_kg : float or None
        Subject mass, kg. Ignored when ``use_weight`` is False.
    img_height, img_width : int
        Per-view target size. The concatenated canvas is ``img_height x (2 * img_width)``.
    threshold : int, default 127
        Foreground threshold; masks are binarised with ``> threshold``.
    use_weight : bool, default True
        When False the weight channel is omitted entirely (V-H), not zero-filled.

    Returns
    -------
    ndarray, shape (3 or 2, img_height, 2 * img_width)
        Channels ``[mask, height, weight]``, float32. Channel 0 is 0/1; channels 1.. are
        constant. ``mask`` is the front and side silhouettes concatenated **horizontally**
        (front left, side right) and resized as a single canvas, so the spatial layout
        preserves the two views side by side.

    Raises
    ------
    ValueError
        If a mask is empty or ``weight_kg`` is None while ``use_weight`` is True.
    """
    if use_weight and weight_kg is None:
        raise ValueError("weight_kg is required when use_weight=True")
    f = (np.asarray(front) > threshold).astype(np.float32)
    s = (np.asarray(side) > threshold).astype(np.float32)
    if not f.any() or not s.any():
        raise ValueError("empty silhouette mask")

    canvas = np.concatenate([f, s], axis=1)
    resized = np.asarray(Image.fromarray((canvas * 255).astype(np.uint8))
                         .resize((2 * img_width, img_height), Image.BILINEAR)) > 127
    mask_ch = resized.astype(np.float32)                 # (H, 2W) - keep it 2-D so that
                                                         # concatenating the (H, 2W) scalar
                                                         # channels below cannot fail on rank

    H, W2 = mask_ch.shape
    chans = [mask_ch, np.full((H, W2), float(height_cm) / HEIGHT_SCALE, np.float32)]
    if use_weight:
        chans.append(np.full((H, W2), float(weight_kg) / WEIGHT_SCALE, np.float32))
    return np.stack(chans, axis=0).astype(np.float32)


class SilhouetteDataset(Dataset):
    """Photo-level dataset yielding assembled CNN tensors and per-subject targets.

    Multiple photos of the same subject appear as separate samples during training. This is
    deliberate: the *split* is by subject (Phase 2), so no subject is ever seen in two splits.

    Parameters
    ----------
    index : BodyMIndex
        Assembled index for the split.
    subject_ids : sequence of str
        Subjects to include. Only their photos are drawn.
    targets : sequence of str
        Target column names.
    img_height, img_width : int
        Per-view target size.
    threshold : int, default 127
        Mask binarisation threshold.
    use_weight : bool, default True
        Include the weight channel.
    augment : callable, optional
        ``augment(front, side) -> (front, side)`` applied to the raw masks before resizing.
    max_photos_per_subject : int, optional
        Cap on photos drawn per subject. Used by ``smoke_test`` so the smoke run touches a
        few hundred samples rather than all 6134 (fix brief section 0).
    """

    def __init__(self, index, subject_ids, targets: Tuple[str, ...] = None,
                 img_height: int = 640, img_width: int = 480, threshold: int = 127,
                 use_weight: bool = True, augment=None,
                 max_photos_per_subject: Optional[int] = None) -> None:
        self.index = index
        self.targets = tuple(targets) if targets is not None else None
        self.img_height = img_height
        self.img_width = img_width
        self.threshold = threshold
        self.use_weight = use_weight
        self.augment = augment
        keep = set(subject_ids)
        photos = index.photos[index.photos.subject_id.isin(keep)]
        if max_photos_per_subject:
            photos = (photos.sort_values("photo_id")
                      .groupby("subject_id", head=0, as_index=False)
                      .head(max_photos_per_subject))
        self.photos = photos.reset_index(drop=True)
        meta = index.subjects.set_index("subject_id")
        self.height = meta["height_cm"].to_dict()
        self.weight = meta["weight_kg"].to_dict()
        self.y = {sid: meta.loc[sid, list(self.targets)].to_numpy(float) for sid in self.height}
        self.subjects = meta

    def photos_per_subject(self) -> pd.Series:
        """Silhouettes available per subject in this dataset.

        Returns
        -------
        pd.Series
            Indexed by ``subject_id``.
        """
        return self.photos.groupby("subject_id").size()

    def __len__(self) -> int:
        """Number of silhouettes (samples)."""
        return len(self.photos)

    def __getitem__(self, i: int):
        """Return ``(tensor, targets, subject_id, photo_id)``.

        Parameters
        ----------
        i : int
            Sample index.

        Returns
        -------
        tensor : Tensor, shape (C, img_height, 2 * img_width)
        targets : ndarray, shape (n_targets,)
        subject_id : str
        photo_id : str
        """
        row = self.photos.iloc[i]
        front = np.array(Image.open(row.front_mask))
        side = np.array(Image.open(row.side_mask))
        if self.augment is not None:
            front, side = self.augment(front, side)
        x = to_tensor(front, side, self.height[row.subject_id], self.weight[row.subject_id],
                      self.img_height, self.img_width, self.threshold, self.use_weight)
        return torch.from_numpy(x), self.y[row.subject_id], row.subject_id, row.photo_id