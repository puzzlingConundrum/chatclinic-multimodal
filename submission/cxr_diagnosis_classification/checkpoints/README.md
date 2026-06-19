# Checkpoint Instructions

## Default Path

Place TorchXRayVision checkpoints here:

```text
chatclinic-multimodal/checkpoints/torchxrayvision/
```

The plugin uses this repo-local path by default. A different path can be configured with:

```bash
export TORCHXRAYVISION_CACHE_DIR=/path/to/chatclinic-multimodal/checkpoints/torchxrayvision
```

## Weights Observed In Local Test

TorchXRayVision downloaded the following file for `densenet121-res224-all`:

```text
nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
```

TorchXRayVision downloaded the following file for `densenet121-res224-chex`:

```text
chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
```

Local file sizes observed:

```text
nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt: 28,382,008 bytes
chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt: 28,381,998 bytes
```

## Full checkpoint set used by the two-stage ensemble

All files live under `chatclinic-multimodal/checkpoints/torchxrayvision/` (~226 MB total — well
under the 3 GB direct-upload limit):

```text
nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-...-best.pt   # densenet121-res224-all (Stage 1 default)
chex-densenet121-...-best.pt                                       # densenet121-res224-chex
nih-densenet121-...-best.pt                                        # densenet121-res224-nih
mimic_ch-densenet121-...-best.pt                                   # densenet121-res224-mimic_ch
mimic_nb-densenet121-...-best.pt                                   # densenet121-res224-mimic_nb
pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt               # resnet50-res512-all (Stage 2)
```

## Stage-2 models without a submitted checkpoint

- **MedCLIP / BiomedCLIP** (`cxr_classifier_tool`): weights auto-download from HuggingFace
  on first use (BiomedCLIP via `open_clip`, CLIP fallback via `transformers`). No `.pt` to submit.
- **LLM escalation router**: uses the OpenAI API (`gpt-5-mini`), not a checkpoint. Set
  `OPENAI_API_KEY` to enable it; otherwise `decision_mode=auto` falls back to the deterministic
  threshold rule, so **inference still runs fully offline without any API key**.

## Submission Note

The `.pt` files are not committed to the repository (`checkpoints/` is in `.gitignore`).

For assignment submission, upload the `.pt` files directly (≤ 3 GB) or provide a Google
Drive/Dropbox link, and state that they must be placed under:

```text
chatclinic-multimodal/checkpoints/torchxrayvision/
```

If the runtime has internet access, TorchXRayVision will also download them automatically on
first inference.
