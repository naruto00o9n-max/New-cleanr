"""Full-resolution page evaluation with versioned local input evidence caching."""
import argparse
import json
from pathlib import Path
import time
from manga_cleaner.config import Settings
from manga_cleaner.evaluation import cache_input_detection
from manga_cleaner.runtime import create_pipeline

parser=argparse.ArgumentParser()
parser.add_argument('image',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();settings=Settings.from_env();data=args.image.read_bytes()
pipeline=create_pipeline(settings);cache_input_detection(pipeline,data);last=0

def progress(stage,done,total):
    global last
    if done==total or time.monotonic()-last>10:
        print(stage,done,total,flush=True);last=time.monotonic()
report=pipeline.clean(data,args.output,'dialogue',progress)
print(json.dumps({k:report[k] for k in ['status','width','height','balloon_count','masked_pixels','punctuation_regions','boundary_repairs','residual_text_count','outside_mask_unchanged','elapsed_seconds']},indent=2),flush=True)
print('Timing includes local cached input evidence when present; use the production CLI for uncached timing.',flush=True)
