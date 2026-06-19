# Demo videos

The presentation slides and demo videos are submitted via the course assignment system
(KLMS), not committed to the repository (to keep the PR lean). Final videos:

| File | Shows |
|------|-------|
| `staged_ui_demo.mp4` | Explicit staged UI: Stage 1 (DenseNet) → press **Run Stage 2 · LLM** / **· Threshold** → Stage 2 result; covers escalate (ambiguous CXR) and single-call (confident CXR), with the LLM-vs-threshold comparison on the same image. |
| `llm_router_demo.mp4` | Auto LLM router on two CXRs: confident → single call, ambiguous → 3-model ensemble, then a grounded `$studio` explanation. |
| `cxr_ensemble_demo.mp4` | Original auto-ensemble flow (pre-staged-UI), for reference. |

Slide deck: `slides/CXR_Ensemble_Presentation.pptx` (13 slides). Slide 11 = results
screenshots, slide 12 = demo video, slide 13 = conclusion.

To regenerate the videos from the running app, see the recorder scripts in the project root
(`record_staged_demo_video.py`, `record_llm_demo_video.py`).
