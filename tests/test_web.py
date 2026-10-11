"""The pages and their script agree about element ids.

There is no JavaScript test runner here and no build step, so a typo in an
element id would otherwise only surface in front of a subject. These checks are
cheap and catch exactly that.
"""

import re

import pytest

from bodymeasure.config import ROOT

WEB = ROOT / "web"
PAGES = ("index.html", "calibrate.html", "review.html")
ID_CALL = re.compile(r'\bel\("([^"]+)"\)')
VIEW_ROW = re.compile(r'li\[data-view="\$\{view\}"\]')


@pytest.fixture(scope="module")
def script() -> str:
    return (WEB / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def markup() -> str:
    return "\n".join((WEB / page).read_text(encoding="utf-8") for page in PAGES)


def test_every_element_the_script_reaches_for_exists(script: str, markup: str) -> None:
    missing = sorted({name for name in ID_CALL.findall(script) if f'id="{name}"' not in markup})
    assert missing == []


def test_every_page_loads_the_shared_script_and_stylesheet() -> None:
    for page in PAGES:
        text = (WEB / page).read_text(encoding="utf-8")
        assert '<script src="/app.js"></script>' in text, page
        assert 'href="/style.css"' in text, page


def test_the_capture_page_has_the_controls_the_brief_requires() -> None:
    page = (WEB / "index.html").read_text(encoding="utf-8")
    for required in (
        'id="consent"',
        'id="subject-id"',
        'id="sex"',
        'id="preview"',
        'id="measure"',
        'data-action="retake"',
    ):
        assert required in page, required
    for view in ("front", "side_left", "side_right"):
        assert f'data-view="{view}"' in page


def test_uploads_are_offered_wherever_the_camera_is(script: str) -> None:
    """The demo must work without the station (brief 6)."""
    page = (WEB / "index.html").read_text(encoding="utf-8")
    assert page.count('type="file"') == 3, "one upload per view"
    assert 'id="upload-frames"' in (WEB / "calibrate.html").read_text(encoding="utf-8")


def test_the_preview_sends_the_unrotated_frame(script: str) -> None:
    """Rotating in the browser as well as on the server would double-rotate."""
    assert "video.videoWidth" in script
    assert "rawFrame" in script


def test_the_review_page_has_what_the_brief_asks_for() -> None:
    page = (WEB / "review.html").read_text(encoding="utf-8")
    for required in ('id="views"', 'id="measurements"', 'id="delete-session"', "download-body"):
        assert required in page, required


def test_dragging_cannot_scroll_the_page_away() -> None:
    """A landmark drag on a touch screen must not pan the page instead."""
    assert "touch-action: none" in (WEB / "style.css").read_text(encoding="utf-8")


def test_deleting_a_session_asks_first(script: str) -> None:
    assert "window.confirm" in script
