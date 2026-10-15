"""Identify processing code and inference dependencies in every page report."""
import hashlib
import importlib.metadata
from pathlib import Path
import json


def engine_fingerprint():
    root=Path(__file__).resolve().parents[1]
    paths=['manga_cleaner/'+name for name in ['quality.py','balloons.py','pipeline.py','models.py','runtime.py','config.py','provenance.py']]+['pyproject.toml']
    digest=hashlib.sha256()
    for name in paths:
        digest.update(name.encode());digest.update((root/name).read_bytes())
    versions={name:importlib.metadata.version(name) for name in ['numpy','opencv-python-headless','torch','torchvision','ultralytics','rapidocr-onnxruntime','onnxruntime','Pillow']}
    digest.update(json.dumps(versions,sort_keys=True).encode())
    return digest.hexdigest()
