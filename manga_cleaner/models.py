"""Pinned model downloads and explicit readiness; no silent fallback."""
import hashlib
from pathlib import Path
import httpx
import numpy as np
import cv2

WEIGHTS = {
    'comictextdetector.pt.onnx': ('https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt.onnx', 'sha256', '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f'),
    'anime-manga-big-lama.pt': ('https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt', 'md5', '29f284f36a0a510bcacf39ecf4c4d54f'),
}

def digest(path, algorithm):
    result = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()

def prepare(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120, trust_env=True) as client:
        for name, (url, algorithm, expected) in WEIGHTS.items():
            path = directory / name
            if path.exists() and digest(path, algorithm) == expected:
                continue
            temporary = path.with_suffix('.partial')
            try:
                with client.stream('GET', url) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_bytes():
                            stream.write(chunk)
                if digest(temporary, algorithm) != expected:
                    raise RuntimeError(f'Checksum mismatch: {name}')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

class Detector:
    def __init__(self, settings):
        self.settings = settings
        cv2.setNumThreads(settings.threads)
        self.net = cv2.dnn.readNetFromONNX(str(settings.models / 'comictextdetector.pt.onnx'))

    def predict(self, rgb):
        h, w = rgb.shape[:2]
        ratio = min(1024 / h, 1024 / w)
        rw, rh = round(w * ratio), round(h * ratio)
        canvas = np.full((1024, 1024, 3), 114, np.uint8)
        canvas[:rh, :rw] = cv2.resize(rgb, (rw, rh))
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255, (1024, 1024), swapRB=False))
        blk, seg = self.net.forward(['blk', 'seg'])
        boxes, scores = [], []
        for row in np.asarray(blk).reshape(-1, 7):
            score = float(row[4] * max(row[5:]))
            cx, cy, bw, bh = row[:4] / ratio
            x0, y0 = max(0, int(cx-bw/2)), max(0, int(cy-bh/2))
            x1, y1 = min(w, int(cx+bw/2)), min(h, int(cy+bh/2))
            if score >= self.settings.confidence and x1-x0 >= 5 and y1-y0 >= 5:
                boxes.append([x0, y0, x1-x0, y1-y0]); scores.append(score)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.settings.confidence, .35)
        regions = [(*boxes[int(i)], scores[int(i)]) for i in np.asarray(indices).reshape(-1)]
        probabilities = cv2.resize(np.asarray(seg)[0, 0, :rh, :rw], (w, h))
        return regions, probabilities >= self.settings.segmentation

class Inpainter:
    def __init__(self, settings):
        import torch
        self.torch = torch
        torch.set_num_threads(settings.threads)
        self.model = torch.jit.load(str(settings.models / 'anime-manga-big-lama.pt'), map_location='cpu').eval()

    def predict(self, rgb, mask):
        torch = self.torch
        h, w = mask.shape
        padding = ((0, (-h) % 8), (0, (-w) % 8))
        image = np.pad(rgb, (*padding, (0, 0)), mode='symmetric')
        binary = np.pad(mask, padding, mode='constant')
        x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()[None] / 255
        m = torch.from_numpy(binary.astype(np.float32))[None, None]
        with torch.inference_mode():
            generated = self.model(x, m)
        return np.clip(generated[0].permute(1, 2, 0).numpy()[:h, :w] * 255, 0, 255).astype(np.uint8)
