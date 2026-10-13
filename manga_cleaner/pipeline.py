"""Page coordinates are authoritative; overlapping tiles never shrink a page."""
import io
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps


def starts(length, size, overlap):
    if size <= overlap or overlap < 0:
        raise ValueError('Invalid tile overlap')
    return sorted(set([*range(0, max(1, length - size + 1), size - overlap), max(0, length - size)]))


def read_image(data, settings):
    if len(data) > settings.max_bytes:
        raise ValueError('Image exceeds byte limit')
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {'PNG', 'JPEG', 'WEBP'} or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Use a static PNG, JPEG or WebP')
        if image.width * image.height > settings.max_pixels:
            raise ValueError('Image exceeds pixel limit; split the page')
        rgb = ImageOps.exif_transpose(image).convert('RGBA')
        background = Image.new('RGBA', rgb.size, 'white')
        background.alpha_composite(rgb)
        return np.asarray(background.convert('RGB')).copy()


def bubble_interior(rgb, text):
    """Accept enclosed flat interiors; never treat a pale face as a white box.

    This conservative guard intentionally leaves SFX/complex bubbles for review.
    Closed neutral or coloured bubbles are found from low-gradient background
    components. Text is temporarily bridged only for topology, not paint masks.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edges = np.maximum(abs(dx), abs(dy)) > 20
    edges = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)).astype(bool)
    # Bridge glyph edges but preserve the outside border of a speech bubble.
    excluded = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    background = (~edges | excluded).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, 8)
    allowed = np.zeros(gray.shape, bool)
    for label in np.unique(labels[text]):
        if label == 0: continue
        x, y, w, h, area = stats[label]
        if area < 80 or area > gray.size * .75 or x == 0 or y == 0 or x+w >= gray.shape[1] or y+h >= gray.shape[0]:
            continue
        component = labels == label
        glyphs = component & text
        if glyphs.sum() < 8:
            continue
        pixels = rgb[component & ~excluded]
        if len(pixels) < 30 or np.percentile(np.max(abs(pixels.astype(float) - np.median(pixels, axis=0)), axis=1), 80) > 18:
            continue
        allowed |= component
    # Fill holes enclosed by an accepted balloon interior. Missing glyph dots
    # are holes in the low-gradient component, not pieces of outside artwork.
    padded = np.pad(allowed.astype(np.uint8), 1)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 1)
    allowed |= flooded[1:-1, 1:-1] == 0
    # Stay inside borders even after glyph mask dilation.
    return cv2.erode(allowed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


class Pipeline:
    def __init__(self, settings, detector, inpainter):
        self.settings, self.detector, self.inpainter = settings, detector, inpainter

    def detect(self, rgb, progress=lambda *args: None):
        s = self.settings
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), bool)
        boxes = []
        windows = [(x, y) for y in starts(h, s.tile, s.overlap) for x in starts(w, s.tile, s.overlap)]
        for index, (x, y) in enumerate(windows):
            crop = rgb[y:y+s.tile, x:x+s.tile]
            regions, segmentation = self.detector.predict(crop)
            if segmentation.shape != crop.shape[:2]:
                raise RuntimeError('Detector returned an invalid segmentation shape')
            accepted = np.zeros(crop.shape[:2], bool)
            for bx, by, bw, bh, confidence in regions:
                accepted[by:by+bh, bx:bx+bw] = True
                boxes.append(dict(x=x+bx, y=y+by, width=bw, height=bh, confidence=round(confidence, 4)))
            mask[y:y+crop.shape[0], x:x+crop.shape[1]] |= segmentation & accepted
            progress('detecting', index+1, len(windows))
        return mask, boxes

    def clean(self, data, output: Path, mode='dialogue', progress=lambda *args: None, manual_mask=None):
        if mode not in {'dialogue', 'all-text'}:
            raise ValueError('Invalid cleaning mode')
        begun = time.monotonic()
        rgb = read_image(data, self.settings)
        text, boxes = self.detect(rgb, progress)
        h, w = text.shape
        interior = np.zeros_like(text)
        if mode == 'dialogue':
            # Guard each window with larger context, then union in page coordinates.
            # The extra context prevents rejection at execution tile seams.
            context = self.settings.tile * 2
            for y in starts(h, context, self.settings.overlap * 2):
                for x in starts(w, context, self.settings.overlap * 2):
                    crop = rgb[y:y+context, x:x+context]
                    interior[y:y+crop.shape[0], x:x+crop.shape[1]] |= bubble_interior(crop, text[y:y+context, x:x+context])
            selected = text & interior
        else:
            selected = text.copy()
        if manual_mask is not None:
            if manual_mask.shape != text.shape:
                raise ValueError('Manual mask dimensions must match oriented image')
            selected |= manual_mask > 0
        diameter = self.settings.dilation * 2 + 1
        mask = cv2.dilate(selected.astype(np.uint8), np.ones((diameter, diameter), np.uint8)).astype(bool)
        if mode == 'dialogue':
            permitted = interior.copy()
            if manual_mask is not None:
                permitted |= cv2.dilate((manual_mask > 0).astype(np.uint8), np.ones((diameter, diameter), np.uint8)) > 0
            mask &= permitted
        result = rgb.copy()
        smooth_mask = mask & interior
        flat_regions = 0
        smooth_regions = 0
        if mode == 'dialogue' and smooth_mask.any():
            # Work on bubble-sized crops. Classical interpolation on smooth
            # balloon backgrounds avoids the neural model's colour halos.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(interior.astype(np.uint8), 8)
            for label in np.unique(labels[smooth_mask]):
                if label == 0: continue
                x, y, bw, bh, _ = stats[label]
                component = labels[y:y+bh, x:x+bw] == label
                local_mask = mask[y:y+bh, x:x+bw] & component
                if not local_mask.any(): continue
                crop = rgb[y:y+bh, x:x+bw]
                safe = component & ~cv2.dilate(local_mask.astype(np.uint8), np.ones((7,7), np.uint8)).astype(bool)
                background = crop[safe]
                if len(background) < 30:
                    smooth_mask[y:y+bh, x:x+bw][component] = False
                    continue
                median = np.median(background, axis=0)
                # Complete glyphs, dots, outlines and shadows inside the verified
                # enclosure, even when segmentation misses fine Arabic strokes.
                contrast = np.max(abs(crop.astype(np.float32)-median), axis=2)
                local_mask |= component & (contrast > 30)
                local_mask = cv2.dilate(local_mask.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool) & component
                mask[y:y+bh, x:x+bw] |= local_mask
                smooth_mask[y:y+bh, x:x+bw] |= local_mask
                target = result[y:y+bh, x:x+bw]
                if np.max(np.std(background.astype(float), axis=0)) <= 4:
                    target[local_mask] = np.round(median).astype(np.uint8)
                    flat_regions += 1
                else:
                    interpolated = cv2.inpaint(crop, local_mask.astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)
                    target[local_mask] = interpolated[local_mask]
                    smooth_regions += 1
        neural_mask = mask & ~smooth_mask
        # Each core is written exactly once; neighbouring context is original.
        size, padding = 768, 128
        cores = [(x, y) for y in range(0, h, size) for x in range(0, w, size)]
        repaired = 0
        for index, (x, y) in enumerate(cores):
            x1, y1 = min(w, x+size), min(h, y+size)
            if neural_mask[y:y1, x:x1].any():
                left, top, right, bottom = max(0, x-padding), max(0, y-padding), min(w, x1+padding), min(h, y1+padding)
                generated = self.inpainter.predict(rgb[top:bottom, left:right], neural_mask[top:bottom, left:right])
                if generated.shape != rgb[top:bottom, left:right].shape:
                    raise RuntimeError('Inpainter returned invalid dimensions')
                if not np.isfinite(generated).all():
                    raise RuntimeError('Inpainter returned non-finite pixels')
                local = generated[y-top:y1-top, x-left:x1-left]
                core_mask = neural_mask[y:y1, x:x1]
                result[y:y1, x:x1][core_mask] = local[core_mask]
                repaired += 1
            progress('cleaning', index+1, len(cores))
        if not np.array_equal(result[~mask], rgb[~mask]):
            raise RuntimeError('Artwork preservation invariant failed')
        after, after_boxes = self.detect(result, lambda _, done, total: progress('verifying', done, total)) if selected.any() else (text, boxes)
        residual = int((after & mask).sum())
        fraction = residual / max(1, int(selected.sum()))
        # Missing or rejected regions are visible in the audit, never a fake success.
        rejected = int((text & ~selected).sum())
        status = 'needs_review' if not selected.any() or fraction > self.settings.residual_fraction or (mode == 'dialogue' and rejected > max(20, text.sum() * .05)) else 'completed'
        report = dict(status=status, mode=mode, width=w, height=h, detected_pixels=int(text.sum()),
                      selected_pixels=int(selected.sum()), masked_pixels=int(mask.sum()), excluded_pixels=rejected,
                      residual_pixels=residual, residual_fraction=round(fraction, 5), repaired_tiles=repaired,
                      flat_regions=flat_regions, smooth_regions=smooth_regions,
                      elapsed_seconds=round(time.monotonic()-begun, 2), boxes=boxes, after_boxes=after_boxes,
                      outside_mask_unchanged=True)
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(output / 'cleaned.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(output / 'mask.png')
        overlay = rgb.copy(); overlay[mask] = (overlay[mask].astype(float)*.35 + np.array([255, 40, 40])*.65).astype(np.uint8)
        Image.fromarray(overlay).save(output / 'overlay.png')
        import json
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return report
