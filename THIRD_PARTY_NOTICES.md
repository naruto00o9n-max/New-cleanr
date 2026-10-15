# Third-party models

The independent orchestration and pipeline code is newly written. Pretrained model weights are external downloads and are not committed.

* Comic Text Detector: https://github.com/zyddnys/manga-image-translator (release beta-0.2.1). Review the upstream GPL-3.0 license and model distribution terms before redistribution or hosted distribution. The SHA-256 of the ONNX artifact is pinned in manga_cleaner/models.py.
* Anime-Manga Big-LaMa: https://github.com/Sanster/models/releases/tag/AnimeMangaInpainting, based on LaMa (https://github.com/advimman/lama). Review the upstream model and repository terms. The published artifact checksum is verified on download.

The originating project inspected during development is https://github.com/k76301992-bit/manga-bubble-cleaner/tree/fix/cleaning-quality-big-lama. Its detector/model artifact locations informed the model adapters; the Expo app, Node processing engine, and Discord bot implementation are not included in this repository.

* YOLO11 Manga109 speech-bubble weights: https://huggingface.co/huyvux3005/manga109-segmentation-bubble, revision f9a4108c4955136a810e5e92207972f3fb3a65fd. The model card declares Apache-2.0. Its upstream dataset/model limitations still apply. The downloaded artifact is pinned by SHA-256.
* Ultralytics 8.3.40: https://github.com/ultralytics/ultralytics, AGPL-3.0 / commercial licensing. The model-card license does not replace the inference library's license.
* RapidOCR ONNX Runtime 1.4.4: https://github.com/RapidAI/RapidOCR, Apache-2.0; includes PP-OCR models in the installed distribution. ONNX Runtime is MIT licensed.
* PanelCleaner and Manga_Cleaner were research references; their source code and repositories are not shipped in this repository. Links and implementation decisions are in docs/quality-research.md.
