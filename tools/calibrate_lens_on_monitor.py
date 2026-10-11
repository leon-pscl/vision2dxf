"""Calibrate the camera lens from a checkerboard shown on the station monitor.

The station has no printed board and no floor mark any more: the image scale comes
from the subject's measured height, so the only thing left to calibrate is the
lens. This tool fills that gap without asking the operator to buy anything.

It draws a checkerboard **on the monitor**, at a known physical size because the
operator types the screen's visible width in millimetres and the script knows the
screen resolution. The board is then shown at a series of positions and rotations
- each is a real pose of a real physical pattern - so ``cv2.calibrateCamera`` has
the varied orientation it needs and writes ``camera_matrix`` and ``dist_coeffs``
into ``config/calibration.yaml``.

**This is a separate milestone from the app**: it is optional, it needs a monitor
and a camera in the same room, and the app runs perfectly well without it
(measurements then carry the ``lens distortion not corrected`` warning).

Run it on the station with nothing else on screen and the room lit evenly::

    python tools/calibrate_lens_on_monitor.py

Press ``q`` or Esc at any point to stop; nothing is written unless the fit
succeeds and the residual is under a pixel.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # run it straight from a checkout
    sys.path.insert(0, str(ROOT / "src"))

from bodymeasure.config import CONFIG_DIR, load_app_config  # noqa: E402

PATTERN = (9, 6)  # inner corners, columns x rows
MIN_FRAMES = 10
MAX_RMS_PX = 1.0
CORNER_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
MARGIN = 0.08  # quiet zone fraction kept clear when drawing
QUIET_PX = 40  # extra pixels of border, so corner detection always has an edge
NOMINAL_SQUARE_MM = 30.0
ANGLES = (-22, -16, -11, -6, 0, 0, 6, 11, 16, 22)
SHIFT = (40, 30)  # pixels the board moves between poses
BLACK, WHITE = 0, 255
COLOUR_CHANNELS = 3  # a BGR frame, as opposed to an already-grey one
FALLBACK_SIZE = (1920, 1080)


@dataclass(frozen=True)
class Intrinsics:
    """What the fit produced, and how well."""

    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    rms_px: float
    frames_used: int


def screen_size() -> tuple[int, int]:
    """Return the primary monitor's pixel size, or a sane default.

    OpenCV cannot ask, so this reads the OS. Getting this right matters: the
    millimetres-per-pixel of the drawn board is the whole basis of the fit.
    """
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()  # otherwise Windows reports a scaled size
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except (AttributeError, OSError):  # pragma: no cover - non-Windows fallback
        return FALLBACK_SIZE


def pitch_px(size: tuple[int, int], monitor_width_mm: float) -> int:
    """Return the board's pixel pitch, shrinking the square until every pose fits.

    30 mm is the nominal square, and what the fit needs is a *known* pitch rather
    than a particular one. The binding constraint is the **tilted** board: a 10x7
    pattern is taller than a 16:9 screen is wide, so fitting it upright would clip
    every rotated pose, and a clipped board has no corners to find.
    """
    width, height = size
    tilt = math.radians(max(abs(angle) for angle in ANGLES))
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    span_x = cols * math.cos(tilt) + rows * math.sin(tilt)
    span_y = cols * math.sin(tilt) + rows * math.cos(tilt)
    room_x = width - 2 * (SHIFT[0] + QUIET_PX)
    room_y = height - 2 * (SHIFT[1] + QUIET_PX)
    # The 0.9 leaves the border corner detection needs, which a board filling the
    # room exactly does not have.
    fits = int(0.9 * min(room_x / span_x, room_y / span_y))
    per_mm = width / monitor_width_mm
    return max(4, min(int(round(NOMINAL_SQUARE_MM * per_mm)), fits))


def square_mm(size: tuple[int, int], monitor_width_mm: float) -> float:
    """Return the physical size of one drawn square, in millimetres."""
    return pitch_px(size, monitor_width_mm) / (size[0] / monitor_width_mm)


def draw_board(
    size: tuple[int, int], pitch: int, angle_deg: float, offset: tuple[int, int]
) -> np.ndarray:
    """Draw the checkerboard rotated by ``angle_deg`` and shifted by ``offset``."""
    width, height = size
    rows, cols = PATTERN[1] + 1, PATTERN[0] + 1
    squares = (np.indices((rows, cols)).sum(axis=0) % 2).astype(np.uint8) * WHITE
    tiles = cv2.resize(squares, (cols * pitch, rows * pitch), interpolation=cv2.INTER_NEAREST)

    centre = (width // 2 + offset[0], height // 2 + offset[1])
    # getRotationMatrix2D already rotates the tile about its own centre; shifting the
    # translation by centre minus half the tile moves that centre on screen without
    # disturbing the rotation.
    matrix = cv2.getRotationMatrix2D(
        (tiles.shape[1] / 2, tiles.shape[0] / 2), angle_deg, 1.0
    )
    matrix[0, 2] += centre[0] - tiles.shape[1] / 2
    matrix[1, 2] += centre[1] - tiles.shape[0] / 2
    warped = cv2.warpAffine(
        tiles, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=WHITE
    )
    return np.where(warped < WHITE, BLACK, WHITE).astype(np.uint8)[:, :, None].repeat(3, axis=2)


def object_points(square: float) -> np.ndarray:
    """Return the board's corner coordinates in mm, on the z = 0 plane."""
    cols, rows = PATTERN
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return grid


def find_corners(image: np.ndarray) -> np.ndarray | None:
    """Return the board's inner corners as ``(N, 2)`` px, or ``None``."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == COLOUR_CHANNELS else image
    found, corners = cv2.findChessboardCornersSB(gray, PATTERN, flags=CORNER_FLAGS)
    return corners.reshape(-1, 2).astype(float) if found else None


def compute_intrinsics(frames: list[np.ndarray], square_mm: float) -> Intrinsics:
    """Fit the camera to several views of the board.

    Raises:
        ValueError: if the frames differ in size, or too few show the whole board.

    """
    sizes = {(frame.shape[1], frame.shape[0]) for frame in frames}
    if len(sizes) > 1:
        raise ValueError(f"frames must all be the same size, got {sorted(sizes)}")

    objects, images = [], []
    for frame in frames:
        corners = find_corners(frame)
        if corners is None:
            continue
        objects.append(object_points(square_mm))
        images.append(corners.astype(np.float32))

    if len(images) < MIN_FRAMES:
        raise ValueError(
            f"found the board in only {len(images)} of {len(frames)} views, need {MIN_FRAMES}"
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objects, images, sizes.pop(), None, None
    )
    return Intrinsics(
        camera_matrix=camera_matrix.tolist(),
        dist_coeffs=dist_coeffs.ravel().tolist(),
        rms_px=float(rms),
        frames_used=len(images),
    )


def write_calibration(intrinsics: Intrinsics, rotation: int, path: Path | None = None) -> Path:
    """Write ``config/calibration.yaml`` and return the path written.

    Only the lens model and the rotation: the file's own mtime is when it was
    calibrated, and the image scale is nowhere near here.
    """
    path = path or CONFIG_DIR / "calibration.yaml"
    payload = {
        "camera_matrix": intrinsics.camera_matrix,
        "dist_coeffs": intrinsics.dist_coeffs,
        "rotation": rotation,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def poses(count: int = 16) -> list[tuple[float, tuple[int, int]]]:
    """Return ``count`` board poses: varied tilt and position.

    A calibration needs orientation variety more than it needs positions, and the
    board has to stay on screen, so the angles stay modest and the offsets small.
    """
    return [
        (float(ANGLES[index % len(ANGLES)]), (40 * (index % 3 - 1), 30 * ((index // 3) % 3 - 1)))
        for index in range(count)
    ]


def capture(
    camera_index: int, size: tuple[int, int], pitch: int, count: int = 16
) -> list[np.ndarray]:
    """Show each pose on the monitor and grab one frame of it."""
    stream = cv2.VideoCapture(camera_index)
    if not stream.isOpened():
        raise RuntimeError(f"camera {camera_index} would not open")
    frames: list[np.ndarray] = []
    try:
        for index, (angle, offset) in enumerate(poses(count), start=1):
            cv2.imshow("calibrate lens", draw_board(size, pitch, angle, offset))
            cv2.waitKey(400)  # let the panel refresh before the frame is taken
            grabbed, frame = stream.read()
            if grabbed:
                frames.append(frame)
            print(f"  view {index:2d}/{count}  {angle:+4.0f} deg", end="\r", flush=True)
    finally:
        stream.release()
        cv2.destroyAllWindows()
    print()
    return frames


def main(argv: list[str]) -> int:
    """Ask for the screen's width, show the board, fit, and save."""
    print(__doc__)
    size = screen_size()
    print(f"Monitor reported at {size[0]}x{size[1]} px.")
    try:
        typed = input(f"Visible width of the screen, in millimetres [{size[0] / 10:.0f}]: ")
        monitor_width_mm = float(typed) if typed.strip() else size[0] / 10.0
        if monitor_width_mm <= 0:
            raise ValueError
    except ValueError:
        print("That is not a width in millimetres. Nothing written.")
        return 1

    pitch = pitch_px(size, monitor_width_mm)
    print(f"Board pitch {pitch} px = {square_mm(size, monitor_width_mm):.1f} mm per square.")
    frames = capture(load_app_config().camera.index, size, pitch)
    # The camera is mounted portrait and the app rotates every frame; the monitor
    # is not rotated, so rotate the captures before fitting.
    rotated = [cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) for frame in frames]
    try:
        intrinsics = compute_intrinsics(rotated, square_mm(size, monitor_width_mm))
    except ValueError as error:
        print(f"Calibration failed: {error}")
        return 1

    print(f"\nrms {intrinsics.rms_px:.3f} px over {intrinsics.frames_used} views")
    if intrinsics.rms_px > MAX_RMS_PX:
        print(f"That is a poor fit (over {MAX_RMS_PX} px). Nothing written: try again.")
        return 1
    print(f"wrote {write_calibration(intrinsics, load_app_config().camera.rotation)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))