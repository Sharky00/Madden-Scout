# Layout Configs

Store one layout file per broadcast style here.

## Current default

- `source_video.json` is calibrated for `output/downloads/source_video.mp4`.

## Why this exists

Different broadcasts and capture pipelines place HUD elements in different areas.
A per-layout config keeps extraction stable without hardcoding one screen position.

## Add a new layout

1. Put a representative local test video in the project.
2. Run discovery and save to a new file:

```bash
python src/video_intake.py "path/to/video.mp4" --output-dir output --sample-every-n-frames 120 --max-frames 20 --save-discovered-layout --layout-config configs/layouts/new_source.json
```

3. Inspect `output/layout_debug/` crops.
4. If needed, hand-edit the `x/y/w/h` values in the new layout file.
5. Use that layout in future runs with `--layout-config`.
