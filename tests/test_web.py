"""The pages and their script agree about element ids, and carry what the brief asks.

There is no JavaScript test runner here and no build step, so a typo in an
element id would otherwise only surface in front of a subject. These checks are
cheap and catch exactly that.
"""

import re

import cv2
import pytest

from bodymeasure.artifacts import STAGES
from bodymeasure.config import ROOT

WEB = ROOT / "web"
PAGES = ("index.html", "process.html", "review.html", "sessions.html")
ID_CALL = re.compile(r'\bel\("([^"]+)"\)')


@pytest.fixture(scope="module")
def script() -> str:
    return (WEB / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def markup() -> str:
    return "\n".join((WEB / page).read_text(encoding="utf-8") for page in PAGES)


@pytest.fixture(scope="module")
def styles() -> str:
    return (WEB / "style.css").read_text(encoding="utf-8")


def test_every_element_the_script_reaches_for_exists(script: str, markup: str) -> None:
    missing = sorted({name for name in ID_CALL.findall(script) if f'id="{name}"' not in markup})
    assert missing == []


def test_every_page_loads_the_shared_script_and_stylesheet() -> None:
    for page in PAGES:
        text = (WEB / page).read_text(encoding="utf-8")
        assert '<script src="/app.js"></script>' in text, page
        assert 'href="/style.css"' in text, page


def test_every_page_has_exactly_one_h1_and_a_language(markup: str) -> None:
    """Brief 16.5: semantic HTML, one logical h1 per page."""
    for page in PAGES:
        text = (WEB / page).read_text(encoding="utf-8")
        assert text.count("<h1") == 1, page
        assert '<html lang="en">' in text, page


def test_the_capture_page_has_the_controls_the_brief_requires() -> None:
    page = (WEB / "index.html").read_text(encoding="utf-8")
    for required in (
        'id="consent"',
        'id="consent-version"',
        'id="subject-id"',
        'id="sex"',
        'id="height-cm"',
        'id="scale-mode"',
        'id="profile"',
        'id="preview"',
        'id="capture-progress"',
        'id="go-process"',
        'data-action="retake"',
    ):
        assert required in page, required
    for view in ("front", "side_left", "side_right"):
        assert f'data-view="{view}"' in page


def test_the_capture_instructions_are_on_screen() -> None:
    """Brief 16.1: what the subject must do, where they can read it."""
    page = (WEB / "index.html").read_text(encoding="utf-8").lower()
    for instruction in (
        "shoes off",
        "hair tied flat",
        "tight cap",
        "fitted clothing",
        "a-pose",
        "uncluttered background",
        "same zoom",
    ):
        assert instruction in page, instruction


def test_the_capture_page_refuses_to_claim_accuracy() -> None:
    """Brief 1: no claim of equivalence to manual anthropometry, no medical use."""
    page = " ".join((WEB / "index.html").read_text(encoding="utf-8").lower().split())
    assert "not been checked against standardised manual anthropometry" in page
    assert "medical" in page


def test_uploads_are_offered_wherever_the_camera_is() -> None:
    """The prototype must work without the station."""
    page = (WEB / "index.html").read_text(encoding="utf-8")
    assert page.count('type="file"') == 3, "one upload per view"
    assert page.count('data-action="sample"') == 3, "one sample per view"


def test_each_view_offers_its_own_stored_sample(script: str) -> None:
    """The button must appear only when the file is there, so the page keeps itself."""
    assert "/samples/${item.dataset.view}.jpg" in script
    assert 'method: "HEAD"' in script
    assert "/samples/${view}.jpg" in script


def test_a_live_capture_is_sent_upright_and_so_is_an_upload(script: str) -> None:
    """The camera is asked for a portrait stream, so nothing has to be turned."""
    assert "height: { ideal: 1920 }" in script
    assert "width: { ideal: 1080 }" in script
    assert "sendView(view, await rawFrame(), isUpright())" in script
    assert "sendView(view, event.target.files[0], true)" in script


def test_capture_runs_every_pose_with_a_three_second_countdown(script: str) -> None:
    """Brief 16.1: a three-second countdown before each capture."""
    page = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="start-capture"' in page
    assert "HOLD_SECONDS = 3" in script
    assert "MOVE_SECONDS = 5" in script
    assert "await countDown(`${name} — hold still`, HOLD_SECONDS)" in script
    assert "flashUntil = performance.now() + FLASH_MS" in script


def test_a_gate_refusal_shows_its_code_and_its_instruction(script: str) -> None:
    """Brief 9.4: the operator is told the code and what to do about it."""
    assert "reason.code}: ${reason.message" in script


def test_the_three_stored_samples_are_present_and_readable() -> None:
    """Dropped in by hand, so pin that what is committed is still usable."""
    samples = WEB / "samples"
    for view in ("front", "side_left", "side_right"):
        path = samples / f"{view}.jpg"
        assert path.exists(), path
        image = cv2.imread(str(path))
        assert image is not None, f"{path} is not a readable image"
        assert image.shape[0] > 200 and image.shape[1] > 200, f"{path} is too small"


def test_the_calibration_page_is_gone(script: str) -> None:
    """The scale is the subject's height, so there is nothing left to calibrate."""
    assert not (WEB / "calibrate.html").exists()
    assert "initCalibrate" not in script


def test_the_stage_viewer_has_every_control_the_brief_lists() -> None:
    """Brief 8.1: previous, next, view and stage selectors, zoom, fit, comparisons."""
    review = (WEB / "review.html").read_text(encoding="utf-8")
    process = (WEB / "process.html").read_text(encoding="utf-8")
    for required in ('id="prev"', 'id="next"', 'id="zoom-in"', 'id="zoom-out"', 'id="fit"',
                     'id="compare"', 'id="blend"', 'id="filmstrip"', 'id="frame"',
                     'id="stage-details"'):
        assert required in review, f"review: {required}"
        assert required in process, f"process: {required}"
    assert 'id="side-by-side"' in review
    assert 'id="view-buttons"' in review
    assert 'id="view-select"' in process


def test_the_processing_page_can_retry_and_cancel() -> None:
    page = (WEB / "process.html").read_text(encoding="utf-8")
    for required in ('id="run"', 'id="retry"', 'id="cancel"', 'id="process-error"'):
        assert required in page, required


def test_the_review_page_has_what_the_brief_asks_for() -> None:
    page = (WEB / "review.html").read_text(encoding="utf-8")
    for required in (
        'id="core"',
        'id="groups"',
        'id="extended"',
        'id="measurements"',
        'id="drawer"',
        'id="export-gate"',
        'id="landmark-select"',
        'id="reset-landmarks"',
        'id="undo"',
        'id="redo"',
        'id="confirm-session"',
        'id="delete-session"',
        'id="download-body"',
        'id="download-extended"',
        'id="download-report"',
        'id="download-csv"',
    ):
        assert required in page, required


def test_the_review_page_offers_every_overlay_and_filter() -> None:
    page = (WEB / "review.html").read_text(encoding="utf-8")
    for overlay in ("mask", "contour", "keypoints", "landmarks", "names", "rows", "values"):
        assert f'data-overlay="{overlay}"' in page, overlay
    for filter_name in ("core", "torso", "arm", "lower", "mesh", "warning", "missing"):
        assert f'data-filter="{filter_name}"' in page, filter_name


def test_the_review_page_separates_confidence_from_validation() -> None:
    """Brief 14: a confident detector is not a validated measurement."""
    page = (WEB / "review.html").read_text(encoding="utf-8")
    script = (WEB / "app.js").read_text(encoding="utf-8")
    assert "Confidence describes how sure the detector was" in page
    assert "Validated against manual measurement" in script


def test_a_typed_value_cannot_be_saved_without_a_note(script: str) -> None:
    """Brief 16.3: an override without a reason is not reviewable."""
    assert "a typed value needs a note" in script
    assert "window.prompt(" in script


def test_landmarks_can_be_moved_from_the_keyboard(script: str) -> None:
    """Brief 16.5: keyboard operation, with an alternative to dragging."""
    for key in ("ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"):
        assert key in script, key
    assert "event.shiftKey ? 10 : 1" in script


def test_the_filmstrip_is_built_from_the_pipeline_itself(script: str) -> None:
    """The UI must not keep its own copy of the stage list."""
    assert "payload.stages" in script
    assert "/artifacts" in script
    for stage in STAGES:
        assert stage.stage_id not in script or stage.stage_id in (
            "00_raw",
            "07_mask_clean",
            "09_edges",
            "10_keypoints",
        ), f"{stage.stage_id} is hard-coded in the script"


def test_images_are_never_stretched(styles: str) -> None:
    """Brief 8.1: the aspect ratio is part of the measurement."""
    assert ".frame img, .frame canvas" in styles
    assert "height: auto" in styles


def test_dragging_cannot_scroll_the_page_away(styles: str) -> None:
    """A landmark drag on a touch screen must not pan the page instead."""
    assert "touch-action: none" in styles


def test_both_themes_come_from_the_same_tokens(styles: str) -> None:
    assert "prefers-color-scheme: dark" in styles
    assert ':root[data-theme="dark"]' in styles
    assert "--accent:" in styles


def test_touch_targets_and_text_sizes_meet_the_brief(styles: str) -> None:
    """Brief 16.5: 44 px targets, 16 px body text, 12 px metadata floor."""
    assert "--touch: 44px" in styles
    assert "min-height: var(--touch)" in styles
    assert "--font: 16px" in styles
    assert "--font-meta: 0.75rem" in styles


def test_motion_and_focus_respect_accessibility(styles: str) -> None:
    assert "prefers-reduced-motion" in styles
    assert ":focus-visible" in styles
    assert "font-variant-numeric: tabular-nums" in styles


def test_status_is_never_colour_alone(styles: str) -> None:
    """Brief 8.1: stage colour must not be the only signal."""
    assert ".state.ok::before" in styles
    assert ".state.bad::before" in styles


def test_a_mobile_width_is_handled(styles: str) -> None:
    assert "max-width: 30rem" in styles
    assert "min-width: 60rem" in styles, "the three-column layout is desktop only"


def test_deleting_a_session_asks_first(script: str) -> None:
    assert "window.confirm" in script
