#!/usr/bin/env python3
"""Web front end for pattern_drafter.

Upload (or paste) vision body measurements, pick a garment, get a drafted
pattern back as JSON and a printable PDF.

    python web.py                                   # http://127.0.0.1:8000
    docker build -t pattern-drafter .
    docker run --rm -p 8000:8000 pattern-drafter

Endpoints
---------
    GET  /                 the single page UI
    GET  /api/garments     template / style / gender catalogue
    POST /api/pattern      {"measurements": "<yaml>", "garment": ..., ...}
    GET  /download/<id>.json, /download/<id>.pdf    last few results
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib.parse import urlparse

from pattern_drafter.drafting import (
    OpenPatternMissing,
    draft_pattern,
    extract_pattern_data,
)
from pattern_drafter.export import export_json, export_pdf
from pattern_drafter.measurements import MeasurementError, TEMPLATES
from pattern_drafter.vision import build_config

INDEX_HTML = Path(__file__).with_name("index.html")

#: How many drafted patterns stay downloadable before the oldest is dropped.
MAX_RESULTS = 20

#: request id -> {"name": str, "json": bytes, "pdf": bytes}
RESULTS: "OrderedDict[str, Dict[str, bytes]]" = OrderedDict()

CONTENT_TYPES = {
    "json": "application/json",
    "pdf": "application/pdf",
}

MAX_UPLOAD = 1 << 20


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------


def catalogue() -> Dict[str, Any]:
    """Garments, styles and genders offered by the UI."""
    return {
        "garments": [
            {
                "id": name,
                "description": spec["description"],
                "supports_darts": spec["supports_darts"],
                "variants": [
                    {
                        "label": "%s — %s"
                        % (style, "woman" if gender == "w" else "man"),
                        "style": style,
                        "gender": gender,
                    }
                    for (style, gender) in sorted(spec["requirements"])
                ],
            }
            for name, spec in sorted(TEMPLATES.items())
        ]
    }


def draft(body: Mapping[str, Any]) -> Dict[str, Any]:
    """Draft one pattern and keep its exports downloadable.

    Returns the JSON response body: the pattern geometry plus the reference
    block that filled in the measurements the body file did not provide.
    """
    try:
        ease = float(body.get("ease", 8.0))
    except (TypeError, ValueError):
        raise MeasurementError("ease must be a number of centimetres")

    config = build_config(
        body.get("measurements") or "",
        template=body.get("garment", ""),
        style=body.get("style", ""),
        gender=body.get("gender", ""),
        ease=ease,
        darts=bool(body.get("darts", False)),
        profile_name="body",
    )

    name = "%s_%s_%s" % (
        config["template"],
        config["style"].lower(),
        config["gender"],
    )

    with tempfile.TemporaryDirectory(prefix="pattern_web_") as tmp:
        # OpenPattern narrates the whole draft on stdout; keep it out of the
        # response and let the container log stay readable.
        with contextlib.redirect_stdout(io.StringIO()):
            pattern = draft_pattern(config)
            data = extract_pattern_data(pattern, config)
            exports = {
                "json": Path(
                    export_json(data, Path(tmp) / "pattern.json")
                ).read_bytes(),
                "pdf": Path(
                    export_pdf(
                        pattern,
                        data,
                        Path(tmp) / "pattern.pdf",
                        {
                            "Pattern": "%s block" % config["template"],
                            "Profile": name,
                        },
                    )
                ).read_bytes(),
            }

    result_id = uuid.uuid4().hex
    RESULTS[result_id] = {"name": name, **exports}
    while len(RESULTS) > MAX_RESULTS:
        RESULTS.popitem(last=False)

    return {
        "id": result_id,
        "name": name,
        "template": config["template"],
        "style": config["style"],
        "gender": config["gender"],
        "reference": config["reference"],
        "scale": config["scale"],
        "from_body": config["from_body"],
        "downloads": {
            fmt: "/download/%s.%s" % (result_id, fmt) for fmt in CONTENT_TYPES
        },
        "pattern": data,
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "pattern_drafter"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            try:
                self._send(200, "text/html; charset=utf-8", INDEX_HTML.read_bytes())
            except OSError as exc:
                self._error(500, "UI file not found: %s" % exc)
            return

        if path == "/api/garments":
            self._json(200, catalogue())
            return

        parts = path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "download":
            result_id, _, fmt = parts[1].partition(".")
            files = RESULTS.get(result_id)
            if files is None or fmt not in CONTENT_TYPES:
                self._error(404, "no such download (results are kept in memory)")
                return
            self._send(
                200,
                CONTENT_TYPES[fmt],
                files[fmt],
                filename="%s.%s" % (files["name"], fmt),
            )
            return

        self._error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if urlparse(self.path).path != "/api/pattern":
            self._error(404, "not found")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(411, "Content-Length required")
            return
        if length > MAX_UPLOAD:
            self._error(413, "upload too large (max %d bytes)" % MAX_UPLOAD)
            return

        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError as exc:
            self._error(400, "invalid JSON body: %s" % exc)
            return
        if not isinstance(payload, dict):
            self._error(400, "body must be a JSON object")
            return

        try:
            self._json(200, draft(payload))
        except MeasurementError as exc:
            self._error(400, str(exc))
        except OpenPatternMissing as exc:
            self._error(503, str(exc))
        except Exception as exc:  # pragma: no cover - drafting library internals
            self.log_error("draft failed: %r", exc)
            self._error(500, "%s: %s" % (type(exc).__name__, exc))

    def _json(self, status: int, payload: Any) -> None:
        self._send(
            status,
            "application/json",
            json.dumps(payload, default=str).encode("utf-8"),
        )

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _send(
        self,
        status: int,
        content_type: str,
        body: bytes,
        filename: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header(
                "Content-Disposition", 'attachment; filename="%s"' % filename
            )
        self.end_headers()
        self.wfile.write(body)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(
        prog="web.py", description="Serve the pattern_drafter web app."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if not INDEX_HTML.is_file():
        parser.error("UI file missing: %s" % INDEX_HTML)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("pattern_drafter web app on http://%s:%d" % (args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())