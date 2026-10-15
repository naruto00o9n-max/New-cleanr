"""Optional evaluation-only input evidence cache; never used by Discord jobs."""
import hashlib
import json
import numpy as np
from .models import WEIGHTS
from .pipeline import Pipeline


def detection_signature(settings):
    return {'model':WEIGHTS['comictextdetector.pt.onnx'][2],
            'tile':settings.tile,'overlap':settings.overlap,
            'confidence':settings.confidence,'segmentation':settings.segmentation,'schema':1}


def cache_input_detection(pipeline, data):
    """Bind one input scan; invalidate evidence after model/settings changes."""
    settings=pipeline.settings
    directory=settings.data/'detection-cache'/hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True,exist_ok=True)
    signature=detection_signature(settings);metadata=directory/'metadata.json'
    original=Pipeline.detect.__get__(pipeline,type(pipeline));first=True
    def detect(rgb,progress=lambda *args:None):
        nonlocal first
        if not first:return original(rgb,progress)
        first=False
        if metadata.exists() and json.loads(metadata.read_text())==signature and (directory/'mask.npy').exists() and (directory/'boxes.json').exists():
            mask=np.load(directory/'mask.npy',allow_pickle=False)
            if mask.shape==rgb.shape[:2] and mask.dtype==bool:
                return mask,json.loads((directory/'boxes.json').read_text())
        mask,boxes=original(rgb,progress)
        np.save(directory/'mask.npy',mask);(directory/'boxes.json').write_text(json.dumps(boxes));metadata.write_text(json.dumps(signature,sort_keys=True))
        return mask,boxes
    pipeline.detect=detect
