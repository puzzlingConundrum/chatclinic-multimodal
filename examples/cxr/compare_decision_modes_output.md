# Stage-2 escalation: hard-coded threshold vs. LLM router

Same DenseNet-121 Stage-1 scores fed to both routers.

| Case | Stage-1 (DenseNet) | Old: threshold rule | New: LLM router | Same outcome? |
|------|--------------------|---------------------|-----------------|---------------|
| Single-call (confident) | Infiltration 0.5251 (margin 0.4454) | escalate=False → single call | escalate=False → single call | ✅ |
| Two-stage (ambiguous) | Cardiomegaly 0.3061 (margin 0.169) | escalate=True → two-stage | escalate=True → two-stage | ✅ |

## LLM reasoning (new router)

- **Single-call (confident)** — Top score 0.5251 for Infiltration with margin 0.4454 over second (0.0797), indicating a clear dominant finding.
- **Two-stage (ambiguous)** — Top score 0.3061 is low (below ~0.35), indicating weak confidence even though the margin to second (0.169) is relatively large.
