# vision2dxf

Made-to-measure garment pattern generation from body photos.

The pipeline has ten steps:

1. Garment type
2. Capture
3. Segmentation
4. Calibration
5. Landmarks
6. Measurement (6a regression, 6b body model)
7. Validation and YAML
8. Drafting
9. Production details
10. Validation (draping and toile)

**Phase 1 is a clickable UI with mock adapters.** Steps 2 to 5 get real
pretrained models in phase 2; everything else stays mocked for now. Every
result carries `is_mock`, and the UI shows a yellow **MOCK DATA – not real
measurements** banner whenever any result in the session is a mock.

## Run it

Two terminals, no containers:

```bash
# backend
cd backend
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt                            # macOS/Linux
.venv/bin/python -m uvicorn main:app --port 8000

# frontend
cd frontend
npm install
npm run dev            # http://localhost:5173, proxies /api to :8000
```

Or with containers, which is the easiest way to get a matching environment on
an unfamiliar machine:

```bash
docker compose up --build
```

Then open http://localhost:5173. The backend is published on :8000 and the
frontend proxies `/api` to it, so both are reachable from the host.

`frontend/Dockerfile` sets `VITE_API_TARGET=http://backend:8000` so the proxy
resolves the service name inside the compose network; on the host, run
`npm run dev` as above and it defaults back to `localhost:8000`.

## Using it

Step 1, pick a garment (Polo shirt / Long sleeve polo / Short sleeve polo /
Slacks). Step 2, either hit **Use sample photos** for a drawn stand-in figure,
or upload your own front and side photos, plus your height. The capture check
then gives you distance, pose, framing and clothing verdicts.

Steps 3 to 5 run off that photo. **Step 5 is where you spend time**: drag the
landmark points onto the right places, then hit Confirm landmarks.

Steps 6 to 10 stay locked until step 5 is confirmed, because everything
downstream reads those landmarks. Steps 7 to 10 additionally wait for step 6 to
have run. After that it is all tables and pattern output.

## Tests

```bash
cd backend  && .venv/bin/python -m pytest -q      # 47 tests, no network, no weights
cd frontend && npm test                           # 12 tests, needs both servers up
npm run build                                     # typecheck plus production build
```

The frontend suite is real coverage, not smoke:

- `api.test.ts` walks all ten steps over HTTP and checks the contracts from the
  client side. Skips if the backend is not on `:8000`.
- `ui.clickthrough.test.ts` drives the actual UI in headless Chromium: it
  uploads both views, reads the capture checklist, drags a landmark and checks
  the point turns `user`, confirms the gate on step 6, runs both backends,
  downloads the YAML, toggles the draft route, and ticks a toile box. Skips if
  either server is down.
- `svg.test.ts` covers the SVG sanitiser.

First run needs the browser: `npx playwright install chromium`.

## Repository layout

```
backend/
  main.py           FastAPI app, in-memory sessions
  contracts.py      frozen Pydantic v2 models, one per step
  registry.py       loads adapters.yaml, instantiates adapters
  adapters.yaml     step -> dotted class path
  config.yaml       every threshold and setting
  adapters/base.py  one Protocol per step
  adapters/mock/    deterministic phase-1 adapters
  adapters/real/    phase 2 lands here
  tests/
frontend/           Vite + React + TypeScript, plain CSS
scripts/            phase 2: weight download and evaluation harness
weights/            git-ignored, populated by scripts/download_weights.py
models/             git-ignored, model packages from the owner's notebook
```

## The core rule

Every model sits behind an adapter that implements a fixed contract. **Swapping
a model takes exactly one edit, to `backend/adapters.yaml`.**

```yaml
segmentation: adapters.mock.segmentation.MockSegmentation
```

Change that one line to `adapters.real.seg_sam2.Sam2Segmenter` and the app
picks it up. No other file changes. This is covered by a test, not just a
convention: `backend/tests/test_registry_swap.py` rewrites the file, clears the
registry cache, and asserts step 3's response changed shape from a different
class while the contract stayed identical.

`contracts.py` is frozen. If a real model cannot satisfy a contract, stop and
report the mismatch rather than widening the contract.

### How to plug in a real model

1. Write the class in `backend/adapters/real/`. It needs a `run` method matching
   the Protocol in `adapters/base.py` and must return the contract model from
   `contracts.py`. Subclass nothing, register nothing.
2. Import it lazily, once per process. Load weights at first `run`, not at
   import, so the app still starts when the weights are missing.
3. Add the import to `backend/requirements.txt` and pin exact versions. A model
   that breaks the whole app on import is worse than no model.
4. Point the step's key in `backend/adapters.yaml` at the new dotted path.
5. Add any thresholds to `backend/config.yaml`. Nothing numeric goes in `.py`.
6. Log inference time on every call.
7. Record the model's licence in `LICENSES_MODELS.md`.
8. Leave the mock registered under a second key so both stay selectable. Do not
   delete the mocks.

`adapters/real/dummy_seg.py` is a worked example: a whole adapter in about 20
lines, used by the swap test.

## Guarantees

**User images are never written to disk.** The browser sends base64, the session
holds it in a dict, and the mask generator builds PNGs in memory. A test
snapshots the repository tree, runs the full pipeline twice, and fails if any
image file appeared. Under Docker the backend container is checked too: it has
no mounts at all, and a full click-through leaves no image file inside it.

**Mocks are deterministic.** Seeded from the input, so the same photo always
gives the same output, which makes the UI testable and screenshots stable.

**Thresholds live in config.** `backend/config.yaml` holds device, pose bands,
pixel error, interval widths, ease limits, seam allowance and the toile
checklist. Change them without touching code.

## Not production-ready

This is a local research prototype. Two things it does not have:

**No authentication.** Anyone who knows a `session_id` can read that session,
including the base64 photos in it. `session_id` is a `uuid4().hex`, so it is not
guessable, but there is no isolation between users. Do not serve this beyond
localhost without adding auth.

**Bounded but still in-memory state.** Sessions live in a dict, capped at
`server.max_sessions` (50) with oldest-first eviction, so a long run cannot
accumulate photos without limit. There is no persistence: restarting the
backend drops every session.

CORS is an allow-list from config, not `*`, so a random website cannot drive a
locally running instance.

## Licence

This is an academic project. Ultralytics YOLO, which phase 2 will use for
detection and segmentation, is licensed **AGPL-3.0**.

Consequence: if this app is served over a network, the full source of the
application must be made available to its users under AGPL-compatible terms.
Internal academic use and evaluation are unaffected.
