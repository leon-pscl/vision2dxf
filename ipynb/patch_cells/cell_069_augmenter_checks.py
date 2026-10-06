def make_production_dataset(split_ids: Sequence[str], use_weight: bool, augment=None):
    """Build the photo-level training dataset for one production variant.

    Parameters
    ----------
    split_ids : sequence of str
        Subject ids for this split.
    use_weight : bool
        True for V-HW, False for V-H. When False the weight channel is dropped by
        :func:`src.bodym_cnn.to_tensor`.
    augment : callable, optional
        ``f(front, side) -> (front, side)`` applied independently per view.

    Returns
    -------
    SilhouetteDataset
        Dataset over the photos of ``split_ids``, capped per subject in a smoke run.
    """
    return SilhouetteDataset(IDX["train"], split_ids, TARGETS,
                             img_height=cfg["img_height"], img_width=cfg["img_width"],
                             threshold=cfg["mask_threshold"], use_weight=use_weight,
                             augment=augment, max_photos_per_subject=SMOKE_PHOTOS)


# --- sanity check: the augmenter must actually change the masks ---------------------------
AUGMENTER, AUG_RNG = training_augmenter(erosion_px=AUG_MAG["erosion_px"],
                                        jitter_sigma_px=AUG_MAG["jitter_sigma_px"],
                                        downsample_factor=AUG_MAG["downsample_factor"],
                                        rotation_deg=AUG_MAG["rotation_deg"],
                                        scale_jitter=AUG_MAG["scale_jitter"],
                                        translate_frac=AUG_MAG["translate_frac"],
                                        seed=cfg["seed"])

_probe_row = IDX["train"].photos.iloc[0]
_probe_f = np.array(Image.open(_probe_row.front_mask))
_probe_s = np.array(Image.open(_probe_row.side_mask))
_probe_out = AUGMENTER(_probe_f.copy(), _probe_s.copy())
_fg_before = (_probe_f > 127).sum()
_fg_after = (_probe_out[0] > 127).sum()
print(f"augmentation check: foreground px {_fg_before} -> {_fg_after} "
      f"({100*(_fg_after-_fg_before)/max(_fg_before,1):+.2f}%), "
      f"views differ independently: {not np.array_equal(_probe_out[0], _probe_out[1])}")
assert (_probe_f > 127).shape == _probe_out[0].shape, "augmentation changed mask shape"
assert _fg_after > 0, "augmentation erased the silhouette"
assert not np.array_equal(_probe_out[0], _probe_out[1]), "views must be perturbed independently"

# --- fix B6: jitter must move the boundary in BOTH directions -------------------------------
_jit_stats = jitter_area_stats(
    [_probe_f] + [np.array(Image.open(r.front_mask))
                  for r in IDX["train"].photos.iloc[1:50].itertuples()],
    sigma_px=max(AUG_MAG["jitter_sigma_px"], 2.0), seed=cfg["seed"])
print(f"jitter area change over {_jit_stats['n']} masks: "
      f"mean {_jit_stats['mean_pct_change']:+.3f}%, "
      f"range [{_jit_stats['min_pct_change']:+.2f}%, {_jit_stats['max_pct_change']:+.2f}%], "
      f"shrank {_jit_stats['n_shrank']}, grew {_jit_stats['n_grew']}")
assert abs(_jit_stats["mean_pct_change"]) < 0.5, (
    f"jitter changes mean foreground area by {_jit_stats['mean_pct_change']:+.3f}%; a symmetric "
    "boundary displacement must be area-neutral on average (fix B6)")
assert _jit_stats["n_shrank"] > 0 and _jit_stats["n_grew"] > 0, (
    f"jitter only moves the boundary one way (shrank {_jit_stats['n_shrank']}, "
    f"grew {_jit_stats['n_grew']}); the previous implementation OR-ed the original foreground "
    "back in, so it could only dilate (fix B6)")

# --- fix B11: DataLoader workers must NOT share one augmentation stream ----------------------
# The test has to hold the *input* constant, otherwise different batches differ for the
# trivial reason that they are different photos. So this wrapper returns the SAME photo for
# every index: any difference between the two workers' outputs can only come from the RNG.
class _SamePhotoDataset:
    """Returns one fixed silhouette pair for every index, so output differences isolate the RNG.

    Parameters
    ----------
    n : int
        Number of identical samples to expose.
    front, side : ndarray
        The fixed masks.
    height_cm, weight_kg : float
        Fixed scalars.
    img_height, img_width, threshold : int
        Preprocessing configuration.
    augment : callable
        Augmentation callable under test.
    """

    def __init__(self, n, front, side, height_cm, weight_kg, img_height, img_width, threshold,
                 augment):
        self.n, self.front, self.side = n, front, side
        self.height_cm, self.weight_kg = height_cm, weight_kg
        self.img_height, self.img_width, self.threshold = img_height, img_width, threshold
        self.augment = augment

    def __len__(self):
        """Number of identical samples."""
        return self.n

    def __getitem__(self, i):
        """Return the augmented tensor for the fixed photo pair.

        Parameters
        ----------
        i : int
            Ignored; every index yields the same photo.

        Returns
        -------
        torch.Tensor
            Assembled tensor.
        """
        f, s = self.augment(self.front.copy(), self.side.copy())
        x = to_tensor(f, s, self.height_cm, self.weight_kg, self.img_height, self.img_width,
                      self.threshold, True)
        return torch.from_numpy(x)


_pm = IDX["train"].subjects.set_index("subject_id").loc[_probe_row.subject_id]
_same_ds = _SamePhotoDataset(8, _probe_f, _probe_s, float(_pm.height_cm), float(_pm.weight_kg),
                             cfg["img_height"], cfg["img_width"], cfg["mask_threshold"], AUGMENTER)
_same_dl = DataLoader(_same_ds, batch_size=4, shuffle=False, num_workers=2,
                      collate_fn=lambda b: b)
_same_batches = list(_same_dl)
assert len(_same_batches) == 2, f"expected 2 batches from 2 workers, got {len(_same_batches)}"
_w0 = np.stack([t.numpy() for t in _same_batches[0]])
_w1 = np.stack([t.numpy() for t in _same_batches[1]])
_same_across_workers = bool(np.array_equal(_w0, _w1))
print(f"2-worker DataLoader, identical input photo in both: "
      f"worker 0 mask pixel counts {_w0[:, 0].reshape(len(_w0), -1).sum(1).tolist()}")
print(f"{'':47s}worker 1 mask pixel counts {_w1[:, 0].reshape(len(_w1), -1).sum(1).tolist()}")
print(f"{'':47s}byte-identical across workers: {_same_across_workers}")
assert not _same_across_workers, (
    "two DataLoader workers given the SAME input photo produced byte-identical augmented "
    "masks, so they share one RNG stream - the augmenter must build its generator lazily per "
    "worker (fix B11)")
print("per-worker RNG: distinct augmentation streams confirmed")

# The streams must also advance between epochs, not repeat.
_e1 = AUGMENTER(_probe_f.copy(), _probe_s.copy())
AUG_RNG.set_epoch(1)
_e2 = AUGMENTER(_probe_f.copy(), _probe_s.copy())
print(f"epoch advance changes the augmentation: {not np.array_equal(_e1[0], _e2[0])}")
assert not np.array_equal(_e1[0], _e2[0]), "AUG_RNG.set_epoch did not change the stream"
print("mark_phase('augmenter checks') =", mark_phase("augmenter checks"), "s")