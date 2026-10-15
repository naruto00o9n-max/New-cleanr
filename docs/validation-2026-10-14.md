# Validation: 2026-10-14–15

The quality corpus contains the two user-supplied long Arabic/English pages and ten additional full-resolution public Asura pages: Myst, Might, Mayhem chapter 119, Nano Machine chapters 1 and 2, and The Greatest Estate Developer chapter 1. Images are identified by URL and SHA-256 in `benchmarks/pages.json`; downloaded images and model weights are not committed.

## Method

There are 97 annotated text/background targets and 33 protected-art regions. All processing uses the same production factory as Discord; no fixture coordinates or image identifiers enter inference. The local iterative run reuses only versioned **input CTD evidence** from the same pinned artifact/settings. Balloon detection, PP-OCR detection, coarse-scale recovery, mask selection, background reconstruction and post-recognition run again. The CI quality gate runs the full input scan without this cache.

The metrics distinguish white paper, dark containers, source-contrast ink coverage, and a grayscale gradient compared with an independently annotated blank strip of the original balloon. Protected regions must remain pixel-identical to the normalized source, as must all pixels outside the final mask. Mask coverage alone does not judge reconstructed texture. Full-page comparisons and difficult cropped regions are also reviewed visually.

The two original user pages and earlier inspected pages are regression/development cases. `estate1-008` and `nano2-005` are additional final validation pages that did not drive the cleaning changes. An earlier nominally held-out smoky page exposed clipped first-line glyphs and was explicitly moved into regression before fixing it. This is a small engineering validation set, not an independent population-wide accuracy estimate.

## Observed failures that became regressions

- The old path only cleared two simple balloons on the user's 900×16000 English page, despite a misleading zero residual from reusing the same detector.
- Large coloured lettering was missed, or a white hole inside a glyph was mistaken for the paper surface.
- One tiny OCR fragment suppressed a multiline Arabic CTD block.
- Dark thought containers, punctuation-only bubbles, and clouds touching panel boundaries were missed.
- Background reconstruction left letter-shaped shadows, and fitting a smooth field across a translucent caption erased the building lines beneath it.
- A newly added dark-background recovery initially regressed the original white GAAH balloon by treating its blue ink as background. The corpus gate caught it; direct paper evidence at the enclosure boundary now determines that polarity.
- White lettering can disappear while its red glow survives neural inpainting. A coloured row/column surface now accepts a majority-black cap when there is sufficient chromatic evidence; a separate red-channel ceiling check detects this previously missed ghost.
- Rectangular model bounds needed explicit zero padding during erosion, and bounding-box deduplication incorrectly discarded small dialogue crossing a larger balloon's frame. Dedicated unit tests cover both failures.
- A grayscale gradient must be preserved, not evaluated against an invented white target. Its gate compares against actual adjacent source-background rows.

## Reproduction and persistence

```bash
python -m pytest tests -q
python scripts/quality_benchmark.py --download --output data/quality
python scripts/quality_benchmark.py --output data/quality-next --compare benchmarks/accepted.json
```

For local iteration, `--cached-detector --resume` may be added. Reuse requires the page's processing fingerprint to match; input detection evidence also carries the artifact/settings signature. The quality workflow fails on failed text/art checks. The comparison additionally fails on removed cases/targets and increased residual or decreased source-ink coverage beyond its small numerical allowance. Code, inference dependency versions and reports are linked by a processing fingerprint.

The unit suite currently has 50 passing tests covering these cleaning failures, tile coverage, exact RGB preservation, model failure, report serialization, ZIP limits, durable queue ownership/recovery and Drive upload retries. Live Discord connection/command registration and Drive delivery are verified separately. A real user slash-command submission is not simulated or claimed by these checks.

Final corpus measurements and the accepted fingerprint are recorded in `benchmarks/accepted.json` after the complete run. Results from an older processing fingerprint cannot pass as a current run. Neither these checks nor the `completed` page status guarantees every possible new image is error-free; unseen layouts can still require review.

## Final measured run

All 12 pages passed all 97 annotated text/background checks and all 33 protected-art checks. Both original pages and both final validation pages passed; all protected regions and all pixels outside the output mask remained unchanged. The final coloured-banner review also confirmed removal of the prominent red letter-shaped glow. Textured reconstruction can still contain fitting seams; this run has no ground-truth clean texture.

| Page | Split | Targets | Protected | Gate |
|---|---|---:|---:|---|
| arabic-original | regression | 9 | 4 | PASS |
| asura-original | regression | 10 | 4 | PASS |
| myst119-02 | regression | 2 | 1 | PASS |
| myst119-03 | development | 12 | 4 | PASS |
| myst119-04 | regression | 4 | 1 | PASS |
| myst119-05 | development | 12 | 3 | PASS |
| myst119-06 | regression | 2 | 1 | PASS |
| nano1-004 | development | 9 | 2 | PASS |
| nano1-008 | development | 4 | 2 | PASS |
| nano1-012 | regression | 8 | 2 | PASS |
| estate1-008 | heldout | 13 | 6 | PASS |
| nano2-005 | heldout | 12 | 3 | PASS |

Processing fingerprint: `c9856998eb948fa086c38618b84100e9c0568861508d78887eb292467ac15aa3`. The recorded run reused versioned input detector evidence; its timing is not an uncached throughput benchmark.
