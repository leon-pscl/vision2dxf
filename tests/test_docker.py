"""The container fetches the YOLO weights itself, and only when they are missing."""

from bodymeasure.config import ROOT

ENTRYPOINT = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_a_present_weight_file_is_left_alone() -> None:
    """A checkout with its own (fine-tuned) weights must not be overwritten."""
    guard = '[ ! -f "/app/models/$name" ]'
    assert guard in ENTRYPOINT
    assert ENTRYPOINT.index(guard) < ENTRYPOINT.index("from ultralytics import YOLO")
    assert 'exec "$@"' in ENTRYPOINT, "the server must still start"


def test_the_image_does_not_bake_the_weights() -> None:
    """The repository is bind-mounted over the image, so a baked copy is hidden."""
    assert "ultralytics import YOLO" not in DOCKERFILE
    assert "COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh" in DOCKERFILE