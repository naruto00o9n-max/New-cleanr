# Quality research and implementation

The 2.1 changes are engineering and evaluation using pretrained weights. No model weights were trained or fine-tuned during this work. Test images are not injected into inference as prompts, masks, coordinates, or per-image branches.

## Primary sources inspected

- [PanelCleaner](https://github.com/VoxelCubes/PanelCleaner): dedicated comic cleaning, mask refinement, connected components and boundary safety. Its GPL-3.0 implementation was inspected for design context; our processing implementation is independently written.
- [Manga_Cleaner](https://github.com/Teemo4621/Manga_Cleaner): speech-bubble segmentation coupled to classical mask refinement rather than assuming text detection alone identifies a safe cleaning surface.
- [Manga109 YOLO11 bubble segmentation weights](https://huggingface.co/huyvux3005/manga109-segmentation-bubble): selected the nano model after comparing it with the [YOLOv8m speech-bubble model](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble) on the user's difficult page. The larger model missed more of its balloons in this experiment. This is a sample-specific model comparison, not a general ranking.
- [Comic Text Detector and manga-image-translator](https://github.com/zyddnys/manga-image-translator): comic-specific text blocks and segmentation.
- [RapidOCR](https://github.com/RapidAI/RapidOCR): a separate PP-OCR detector and recognizer. Detection is retained even when stylized/Arabic text cannot be recognized.
- [Anime-Manga Big-LaMa](https://github.com/Sanster/models/releases/tag/AnimeMangaInpainting): background reconstruction where flat paper or a fitted background is insufficient.

## Changes that addressed observed failures

1. Combine independently proposed text and balloon evidence; maintain multiline CTD support even if OCR only finds one tiny fragment.
2. Tile in both axes with overlap and retain the full source resolution. Very large unresolved text proposals trigger a second detector scale; a whole long chapter is never squeezed into one detector input.
3. Reject weak balloon proposals unless independent comic-text evidence plus readable text, or a paired punctuation pattern, confirms them. The proposal threshold is not the acceptance threshold.
4. Complete letter interiors, outlines, antialiasing and shadows. A white hole inside a giant letter must not be mistaken for the entire bubble's white paper.
5. Recover grayscale thoughts on both black and light backgrounds, including containers overlapping panel boundaries; extend neutral support when a learned polygon clips a confirmed line.
6. Recover isolated multiline captions on white paper. Protect small dense interface-label clusters and require independent evidence before inventing a narration enclosure.
7. Reconstruct flat paper and verified smooth backgrounds directly. Translucent captions need neural reconstruction so underlying panel lines survive. Large coloured gradient/striped containers use background evidence from their colour field rather than treating white lettering as white paper.
8. Independently recognize treated crops afterwards, preserve exact RGB outside the final mask, and expose excluded/uncertain regions. Neither OCR nor mask coverage proves that every human-visible glyph vanished.

## Permanent quality gate

`benchmarks/pages.json` stores source URLs, SHA-256 values, source-image annotations and protected art. The images themselves are not committed. `scripts/quality_benchmark.py` renders the complete pages with the production factory, checks the annotations and rejects any previously passing target that fails when `--compare` is supplied. It also checks protected-art pixels and records the engine fingerprint.

The initial two user pages are regression cases. Additional pages from public Asura chapter readers cover long action pages, connected speech balloons, black/red thoughts, giant display lettering, punctuation-only dialogue, gradients and small dialogue beside a portrait. A held-out page that exposes a failure is moved into regression before fixing it; it is not subsequently counted as untouched validation. Textured fills and reconstructed frames are also visually reviewed, because numerical coverage alone cannot judge invented background texture.
