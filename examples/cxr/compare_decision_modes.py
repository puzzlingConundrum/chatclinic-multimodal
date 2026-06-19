"""Side-by-side comparison of the OLD hard-coded threshold router vs. the NEW
LLM router for the two-stage CXR ensemble.

Runs both decision modes on the two demo images (one confident -> single call,
one ambiguous -> two-stage ensemble) and writes a markdown table for the report.

Usage (from chatclinic-multimodal/):
    /home/jinotter3/miniconda3/envs/med/bin/python examples/cxr/compare_decision_modes.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # chatclinic-multimodal/
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# load .env so OPENAI_API_KEY is visible (same as the FastAPI app does)
envp = ROOT / ".env"
if envp.exists():
    for line in envp.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            os.environ.setdefault(k, v)
os.environ.setdefault("TORCHXRAYVISION_CACHE_DIR", str(ROOT / "checkpoints/torchxrayvision"))

from plugins.cxr_ensemble_tool.logic import run_ensemble  # noqa: E402

CASES = [
    ("Single-call (confident)", "examples/cxr/demo_cases/single_call_confident.png"),
    ("Two-stage (ambiguous)",   "examples/cxr/demo_cases/two_stage_ambiguous.png"),
]

rows = []
for label, img in CASES:
    out = {"label": label, "image": Path(img).name}
    for mode in ("threshold", "llm"):
        r = run_ensemble(img, "image", decision_mode=mode)
        d = r.get("decision", {})
        rc = d.get("raw_confidence", {})
        out[mode] = {
            "stage1": f"{rc.get('top_finding')} {rc.get('top_score')} (margin {rc.get('margin')})",
            "escalate": bool(d.get("escalate")),
            "reason": d.get("reason", ""),
            "triggered": bool(r.get("ensemble_triggered")),
            "models": r.get("models_used", []),
            "decided_by": r.get("decided_by"),
        }
    rows.append(out)

# ---- console ----
for o in rows:
    print(f"\n### {o['label']}  —  {o['image']}")
    print(f"  Stage-1 DenseNet: {o['threshold']['stage1']}")
    for mode in ("threshold", "llm"):
        m = o[mode]
        call = "TWO-STAGE (3 models)" if m["triggered"] else "SINGLE CALL (DenseNet only)"
        print(f"  [{mode:9}] escalate={str(m['escalate']):5} -> {call}")
        print(f"             reason: {m['reason']}")

# ---- markdown for the report ----
md = ["# Stage-2 escalation: hard-coded threshold vs. LLM router\n",
      "Same DenseNet-121 Stage-1 scores fed to both routers.\n",
      "| Case | Stage-1 (DenseNet) | Old: threshold rule | New: LLM router | Same outcome? |",
      "|------|--------------------|---------------------|-----------------|---------------|"]
for o in rows:
    t, l = o["threshold"], o["llm"]
    t_call = "two-stage" if t["triggered"] else "single call"
    l_call = "two-stage" if l["triggered"] else "single call"
    same = "✅" if t["triggered"] == l["triggered"] else "❌ differs"
    md.append(f"| {o['label']} | {t['stage1']} | escalate={t['escalate']} → {t_call} "
              f"| escalate={l['escalate']} → {l_call} | {same} |")
md.append("\n## LLM reasoning (new router)\n")
for o in rows:
    md.append(f"- **{o['label']}** — {o['llm']['reason']}")
out_md = ROOT / "examples/cxr/compare_decision_modes_output.md"
out_md.write_text("\n".join(md) + "\n")
print(f"\nWrote {out_md}")
