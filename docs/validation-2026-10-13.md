# Validation: 2026-10-13

The engine was evaluated on the user-provided real Arabic manhwa page, 800×13855 pixels, at full resolution.

## Evidence

- 29 automated tests passed: long pages across both tile axes, seam-crossing glyphs, final rows, exact preservation outside the mask, model failures, empty detection, residual text, closed bubbles, small missed glyphs, pale text outlines, smooth gradients, UI-cluster exclusion with overlap duplicates, queue recovery and ownership, cancellation, upload retries, Drive chunk acknowledgments, ZIP bounds and natural ordering, and separation of finished pages from audit images.
- Discord REST authentication succeeded. The ZEUS Cl bot connected to the gateway, and four slash commands were registered: clean, status, cancel, retry_delivery.
- Google Drive OAuth succeeded. A real output folder is created and uploaded during validation; its link is returned to the user separately.
- Model artifact checksums verified for the ONNX comic text detector and Anime-Manga Big-LaMa.

## Full-page result

- Dimensions: 800×13855, unchanged.
- Detected text pixels: 195250.
- Selected text pixels: 151920.
- Final mask pixels: 271524.
- Excluded text pixels: 43330.
- Protected interface text pixels: 10507.
- Residual detector positives inside the treated regions: 0.
- Repair paths: 1 uniform background, 7 fitted smooth gradient, 1 Telea, 0 neural tiles.
- Exact RGB equality outside the final mask was verified.
- Final processing plus verification took 56.65 seconds **with cached input detection**. A separate initial input scan took approximately 99 seconds on this CPU. These are one-page measurements, not a throughput guarantee.

Full-page previews and full-resolution paired crops were inspected. A first neural-only implementation left dark or pale glyph residues. Smooth gradient reconstruction alone also left pale outlines that the neural detector did not flag. The final pipeline bridges outlines during enclosure analysis and masks deviations from the fitted colour field near detected text. The interface behind the female character is protected by excluding dense tiny-label clusters.

The output remains `needs_review`: free text outside accepted bubbles, some signage, and deliberately preserved interface labels remain. This is a conservative dialogue-mode result, **not** proof that every text element on the page was removed. Dialogue-only scope is heuristic; a broad labeled benchmark and further semantic balloon classification are still needed.

No real user slash-command chapter submission was available during this session. Gateway connection and registration are verified separately; processing and live Drive upload use the same engine and delivery adapter. Drive-source imports and resumable retry edge cases are covered by mocked REST tests.

## Reproduction

```bash
python -m pytest tests -q
manhwa-cleaner prepare-models
manhwa-cleaner doctor
python scripts/evaluate_page.py page.jpg --output data/evaluation
```

Local fixture and generated images are ignored by Git. Model weights and credentials are also local only.
