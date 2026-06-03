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

## Submission Note

Checkpoint files are not committed to the repository because they are downloaded runtime assets.

For assignment submission, upload the required `.pt` files separately or provide a Google Drive/Dropbox link, and state that they should be placed under:

```text
chatclinic-multimodal/checkpoints/torchxrayvision/
```

If the runtime has internet access, the file can also be downloaded automatically on first inference by TorchXRayVision.
