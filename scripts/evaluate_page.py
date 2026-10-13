"""Reproducible, full-resolution evaluation; caches detector evidence locally."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from manga_cleaner.config import Settings
from manga_cleaner.models import Detector, Inpainter
from manga_cleaner.pipeline import Pipeline, read_image

parser=argparse.ArgumentParser()
parser.add_argument('image',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
settings=Settings.from_env()
args.output.mkdir(parents=True,exist_ok=True)
data=args.image.read_bytes()
key=hashlib.sha256(data).hexdigest()
cache=settings.data/'detection-cache'/key
cache.mkdir(parents=True,exist_ok=True)
pipeline=Pipeline(settings,Detector(settings),Inpainter(settings))
original=pipeline.detect
first=True

def detect(rgb,progress=lambda *args: None):
    global first
    if first:
        first=False
        if (cache/'mask.npy').exists() and (cache/'boxes.json').exists():
            print('Using cached input detection',flush=True)
            return np.load(cache/'mask.npy'),json.loads((cache/'boxes.json').read_text())
        mask,boxes=original(rgb,progress)
        np.save(cache/'mask.npy',mask)
        (cache/'boxes.json').write_text(json.dumps(boxes))
        return mask,boxes
    return original(rgb,progress)

pipeline.detect=detect
last=0

def progress(stage,done,total):
    global last
    if done == total or time.monotonic()-last>10:
        print(stage,done,total,flush=True);last=time.monotonic()
report=pipeline.clean(data,args.output,'dialogue',progress)
print(json.dumps({k:v for k,v in report.items() if k not in {'boxes','after_boxes'}},indent=2),flush=True)
