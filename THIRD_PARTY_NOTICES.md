# Third-party models

The independent orchestration and pipeline code is newly written. Pretrained model weights are external downloads and are not committed.

* Comic Text Detector: https://github.com/zyddnys/manga-image-translator (release beta-0.2.1). Review the upstream GPL-3.0 license and model distribution terms before redistribution or hosted distribution. The SHA-256 of the ONNX artifact is pinned in manga_cleaner/models.py.
* Anime-Manga Big-LaMa: https://github.com/Sanster/models/releases/tag/AnimeMangaInpainting, based on LaMa (https://github.com/advimman/lama). Review the upstream model and repository terms. The published artifact checksum is verified on download.

The originating project inspected during development is https://github.com/k76301992-bit/manga-bubble-cleaner/tree/fix/cleaning-quality-big-lama. Its detector/model artifact locations informed the model adapters; the Expo app, Node processing engine, and Discord bot implementation are not included in this repository.
