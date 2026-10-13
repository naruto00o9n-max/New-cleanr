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
        if image.width > settings.max_width or image.height > settings.max_height:
            raise ValueError('Image dimensions exceed limits; split the page')
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
    excluded = cv2.dilate(text.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
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
        gy, gx = np.nonzero(glyphs)
        # UI headers attached to the edge of a panel should not masquerade
        # as centered speech/narration text in a closed balloon.
        if abs((gx.min()+gx.max())/2 - (x+w/2)) > w*.30 or abs((gy.min()+gy.max())/2 - (y+h/2)) > h*.30:
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


def ui_protection(boxes, height, width):
    """Conservatively preserve dense clusters of tiny interface labels.

    Duplicate detections at overlapping tile seams must count only once.
    This is an exclusion heuristic, not a semantic balloon classifier.
    """
    small = []
    for box in boxes:
        if box['height'] > width*.05: continue
        if any(abs(box['x']-b['x'])<12 and abs(box['y']-b['y'])<12 for b in small): continue
        small.append(box)
    protected = np.zeros((height, width), bool)
    for box in small:
        group = [b for b in small if abs(b['y']-box['y']) < 500]
        if len(group) < 4: continue
        x0 = max(0,min(b['x'] for b in group)-24)
        y0 = max(0,min(b['y'] for b in group)-24)
        x1 = min(width,max(b['x']+b['width'] for b in group)+24)
        y1 = min(height,max(b['y']+b['height'] for b in group)+160)
        protected[y0:y1,x0:x1] = True
    return protected


def smooth_background(crop, safe):
    """Robust quadratic colour field for smooth balloon gradients.

    Fit only unmasked background samples. This avoids Telea copying pale glyph
    halos back into the hole and preserves gentle shading better than white fill.
    Return None if a smooth surface cannot explain the observed background.
    """
    h, w = safe.shape
    indices = np.flatnonzero(safe)
    if len(indices) < 40: return None
    indices = indices[::max(1, len(indices)//4096)]
    ys, xs = np.unravel_index(indices, safe.shape)
    def design(x, y):
        x = x / max(1, w-1) * 2-1; y = y / max(1, h-1) * 2-1
        return np.stack((np.ones_like(x), x, y, x*y, x*x, y*y), axis=-1)
    features = design(xs.astype(float), ys.astype(float))
    values = crop.reshape(-1,3)[indices].astype(float)
    keep = np.ones(len(indices), bool)
    for _ in range(3):
        coefficients, _, rank, _ = np.linalg.lstsq(features[keep], values[keep], rcond=None)
        if rank < 6: return None
        error = np.max(abs(features @ coefficients - values), axis=1)
        keep = error <= max(4, np.percentile(error, 85))
        if keep.sum() < 40: return None
    if np.percentile(error, 85) > 10: return None
    yy, xx = np.mgrid[:h, :w]
    return np.clip(design(xx.astype(float), yy.astype(float)) @ coefficients, 0, 255).round().astype(np.uint8)


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
        support = np.zeros_like(text)
        for box in boxes:
            x, y, bw, bh = box['x'], box['y'], box['width'], box['height']
            support[max(0,y-12):min(h,y+bh+12),max(0,x-12):min(w,x+bw+12)] = True
        protected = ui_protection(boxes,h,w) if mode == 'dialogue' else np.zeros_like(text)
        guard_text = text & ~protected
        if mode == 'dialogue':
            # Guard each window with larger context, then union in page coordinates.
            # The extra context prevents rejection at execution tile seams.
            context = self.settings.tile * 2
            for y in starts(h, context, self.settings.overlap * 2):
                for x in starts(w, context, self.settings.overlap * 2):
                    crop = rgb[y:y+context, x:x+context]
                    interior[y:y+crop.shape[0], x:x+crop.shape[1]] |= bubble_interior(crop, guard_text[y:y+context, x:x+context])
            interior &= ~protected
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
        gradient_regions = 0
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
                # Compare to the fitted background, not just to darkness.
                # White outlines can differ from a pale balloon by only 8–15
                # levels and are commonly missed by the learned segmentation.
                is_flat = np.max(np.std(background.astype(float), axis=0)) <= 4
                field = None if is_flat else smooth_background(crop, safe)
                expected = median if field is None else field
                contrast = np.max(abs(crop.astype(np.float32)-expected), axis=2)
                local_mask |= component & support[y:y+bh,x:x+bw] & (contrast > (6 if is_flat or field is not None else 20))
                local_mask = cv2.dilate(local_mask.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool) & component
                mask[y:y+bh, x:x+bw] |= local_mask
                smooth_mask[y:y+bh, x:x+bw] |= local_mask
                target = result[y:y+bh, x:x+bw]
                if is_flat:
                    target[local_mask] = np.round(median).astype(np.uint8)
                    flat_regions += 1
                else:
                    if field is not None:
                        target[local_mask] = field[local_mask]
                        gradient_regions += 1
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
                      flat_regions=flat_regions, smooth_regions=smooth_regions, gradient_regions=gradient_regions,
                      elapsed_seconds=round(time.monotonic()-begun, 2), boxes=boxes, after_boxes=after_boxes,
                      outside_mask_unchanged=True, protected_ui_pixels=int((protected & text).sum()))
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(output / 'cleaned.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(output / 'mask.png')
        overlay = rgb.copy(); overlay[mask] = (overlay[mask].astype(float)*.35 + np.array([255, 40, 40])*.65).astype(np.uint8)
        Image.fromarray(overlay).save(output / 'overlay.png')
        import json
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return report
