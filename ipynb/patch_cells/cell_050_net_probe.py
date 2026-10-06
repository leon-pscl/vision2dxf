# --- instantiate and sanity-check the network --------------------------------------------
from bodym_cnn import (HEIGHT_SCALE, N_CHANNELS, N_CHANNELS_VH, WEIGHT_SCALE, MeasurementNet,
                       SilhouetteDataset, build_backbone, to_tensor)

# fix A2: prove the configured backbone actually exists in this torchvision build, and print
# the available MNASNet names if it does not. The previous default "mnasnet1_0_3_0" is not a
# torchvision model, so build_backbone raised RuntimeError on every run.
import torchvision.models as _tvm

_avail_mnas = sorted(n for n in dir(_tvm)
                     if n.startswith("mnasnet") and len(n) > 7 and n[7].isdigit())
print("configured backbone:", cfg["backbone"])
print("available torchvision MNASNet models:", _avail_mnas)
assert hasattr(_tvm, cfg["backbone"]), (
    f"cfg['backbone'] = {cfg['backbone']!r} is not in torchvision. Available MNASNet: "
    f"{_avail_mnas} (fix A2)")
assert cfg["backbone"] in _avail_mnas, (
    f"cfg['backbone'] = {cfg['backbone']!r} is not an MNASNet; expected one of {_avail_mnas}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| GPU:", ENV["gpu"]["name"])

net_b2 = MeasurementNet(backbone=cfg["backbone"], pretrained=cfg["pretrained"],
                        n_outputs=len(TARGETS), hidden=cfg["hidden"], dropout=cfg["dropout"],
                        use_weight=True, offline_weights=cfg["offline_weights"]).to(device)
print("backbone weights source:", net_b2.weights_source)
print(f"input channels: V-HW {N_CHANNELS}, V-H {N_CHANNELS_VH} -> head outputs: {net_b2.n_outputs}")
print(f"per-view size {cfg['img_height']}x{cfg['img_width']} "
      f"(canvas {cfg['img_height']}x{2*cfg['img_width']}), batch {cfg['batch_size']}")

net_b2.eval()
with torch.no_grad():
    x_dummy = torch.zeros(2, N_CHANNELS, cfg["img_height"], 2 * cfg["img_width"], device=device)
    y_dummy = net_b2(x_dummy)
print("forward shape:", tuple(y_dummy.shape), "| finite:", bool(torch.isfinite(y_dummy).all()))
assert y_dummy.shape == (2, len(TARGETS)) and torch.isfinite(y_dummy).all()
print("stem adaptation:", net_b2.stem_adaptation,
      "| trunk input channels:", net_b2.stem_in_channels())
assert net_b2.stem_in_channels() == 1, (
    f"the mask is one channel; the adapted stem should be 1-in, got "
    f"{net_b2.stem_in_channels()}. Expected the mean-reduced grayscale stem (fix D3).")
print("  note: torchvision exposes the trunk as 'features' on ResNet-family models but as "
      "'layers' on MNASNet, which has no '.features' attribute at all.")

# fix B2: the per-view size must preserve the native mask aspect ratio, or the CNN is trained
# on squashed people. BodyM masks are portrait 4:3, so H > W.
_native = EDA["image_stats"][["height_px", "width_px"]].drop_duplicates()
_native_ar = (_native.height_px / _native.width_px).to_numpy()
_model_ar = cfg["img_height"] / cfg["img_width"]
print(f"native mask sizes: {list(_native.itertuples(index=False, name=None))}")
print(f"native aspect (H/W): {np.round(_native_ar, 4).tolist()}")
print(f"model per-view aspect (H/W): {_model_ar:.4f}  ({cfg['img_height']}x{cfg['img_width']})")
_ar_err = np.abs(_native_ar - _model_ar) / _native_ar
print(f"relative aspect error: {np.round(_ar_err, 5).tolist()}")
assert (_ar_err < 0.01).all(), (
    f"model per-view aspect {_model_ar:.4f} differs from the native mask aspect by more than "
    f"1% ({np.round(_ar_err*100, 2).tolist()}%). BodyM masks are portrait; set "
    f"img_height/img_width to preserve the ratio (fix B2).")

# fix B1: the scalars now reach the head's first Linear directly. The old test asserted the
# output was IDENTICAL when only the scalar channels changed (a zero-init property); that test
# is deleted because the property it pinned was precisely the bug. The replacement checks are:
#   (a) the scalar readout is exact,
#   (b) a change in the scalar channels DOES change the output at initialisation.
net_b2.eval()
with torch.no_grad():
    a = torch.rand(1, N_CHANNELS, cfg["img_height"], 2 * cfg["img_width"], device=device)
    a[:, 1] = 0.86     # height channel
    a[:, 2] = 0.74     # weight channel
    b = a.clone()
    b[:, 1] = 0.86 + 0.05
    b[:, 2] = 0.74 - 0.05
    sc_a, sc_b = net_b2.scalar_channels(a), net_b2.scalar_channels(b)
    print(f"scalar readout at init: A = {sc_a.cpu().numpy().ravel().round(4).tolist()}")
    print(f"                       B = {sc_b.cpu().numpy().tolist()}")
    assert torch.allclose(sc_a[:, 0:1], a[:, 1].mean(dim=(1, 2)).unsqueeze(1)), \
        "scalar_channels must read the height channel exactly"
    delta = (net_b2(a) - net_b2(b)).abs().max().item()
print(f"max |delta output| when only height/weight channels change, at init: {delta:.3e}")
assert delta > 1e-6, (
    "changing height/weight does not change the output at initialisation, so the scalar path "
    "is dead again (fix B1). The previous zero-initialised projection could never learn.")

# A wrong channel count must be rejected loudly rather than silently mis-indexed.
try:
    net_b2(torch.zeros(1, N_CHANNELS_VH, cfg["img_height"], 2 * cfg["img_width"], device=device))
    raise AssertionError("MeasurementNet accepted a 3-channel input with use_weight=True")
except ValueError:
    print(f"channel-count guard: {N_CHANNELS_VH}-channel input rejected by V-HW, as intended")

# fix B11: the head's input-side weight columns are the only route from scalars to predictions.
with torch.no_grad():
    _w = net_b2.head[0].weight
    print(f"head input width {int(_w.shape[1])} = feat_dim {net_b2.feat_dim} "
          f"+ n_scalars {net_b2.n_scalars}")
    print(f"head weight column norms: image features {float(_w[:, :net_b2.feat_dim].norm()):.4f}, "
          f"scalars {float(_w[:, net_b2.feat_dim:].norm()):.4f}")
    assert int(_w.shape[1]) == net_b2.feat_dim + net_b2.n_scalars, \
        "head input width must be feat_dim + n_scalars (fix B1)"

n_params = sum(p.numel() for p in net_b2.parameters())
print(f"trainable parameters: {n_params/1e6:.2f} M")
print("mark_phase('B2 model') =", mark_phase("B2 model"), "s")