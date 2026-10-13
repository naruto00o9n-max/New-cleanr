from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    data: Path = ROOT / 'data'
    models: Path = ROOT / 'models'
    tile: int = 1024
    overlap: int = 256
    confidence: float = .35
    segmentation: float = .30
    dilation: int = 2
    max_width: int = 8192
    max_height: int = 100_000
    max_pixels: int = 40_000_000
    max_bytes: int = 50 * 1024 * 1024
    max_pages: int = 100
    max_batch_bytes: int = 300 * 1024 * 1024
    queue_limit: int = 12
    residual_fraction: float = .03
    threads: int = 2

    @classmethod
    def from_env(cls):
        return cls(data=Path(os.getenv('CLEANER_DATA_DIR', str(ROOT / 'data'))),
                   models=Path(os.getenv('CLEANER_MODELS_DIR', str(ROOT / 'models'))),
                   threads=int(os.getenv('CLEANER_THREADS', '2')))
