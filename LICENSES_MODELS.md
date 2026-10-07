# Model licences

One row per model the project will use. Phase 1 runs mocks only, so this table
is the plan of record for phase 2. Fill in the version column as weights get
pinned; `scripts/download_weights.py` will read these URLs.

| Model | Version | Licence | URL |
|---|---|---|---|
| RTMPose (mmpose / OpenMMLab) | TODO: pin at download | Apache-2.0 | https://github.com/open-mmlab/mmpose |
| SAM 2 (Meta) | TODO: pin at download | Apache-2.0 | https://github.com/facebookresearch/sam2 |
| SAM 3 (Meta) | TODO: pin at download | see repo, confirm before use | https://github.com/facebookresearch/sam3 |
| Ultralytics YOLO (detect + seg) | TODO: pin at download | AGPL-3.0 | https://github.com/ultralytics/ultralytics |

## Notes

- Ultralytics YOLO ships under **AGPL-3.0**. That is the licence that shapes how
  this app may be distributed.
- SAM 3's licence has not been confirmed for this project yet. Check the repo
  before wiring `adapters/real/seg_sam3.py`. If its terms are incompatible with
  academic redistribution, use `seg_sam2.py` or `seg_yolo.py` instead.
- Every downloaded checkpoint must have its checksum recorded in
  `scripts/download_weights.py` before it is committed. `/weights` is
  git-ignored.
