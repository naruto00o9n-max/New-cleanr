from dataclasses import replace
import io
import json
import numpy as np
from PIL import Image, ImageDraw
import pytest
from manga_cleaner.config import Settings
from manga_cleaner.pipeline import Pipeline, starts, read_image, bubble_interior

class Detector:
    def predict(self, rgb):
        mask = (rgb[:,:,0] < 30) & (rgb[:,:,1] < 30)
        return [(0,0,rgb.shape[1],rgb.shape[0],.9)], mask

class Painter:
    def predict(self, rgb, mask):
        return np.full_like(rgb, 255)

def png(rgb):
    stream = io.BytesIO(); Image.fromarray(rgb).save(stream, format='PNG'); return stream.getvalue()

@pytest.mark.parametrize('length', [1, 1024, 1025, 5000, 13855])
def test_tiles_cover_every_pixel(length):
    covered = np.zeros(length, bool)
    for position in starts(length, 1024, 256): covered[position:position+1024] = True
    assert covered.all()


def test_long_page_seams_and_exact_art_preservation(tmp_path):
    rng = np.random.default_rng(13)
    rgb = rng.integers(80, 230, (4200, 1200, 3), dtype=np.uint8)
    rgb[760:790, 740:790] = 0  # crosses inpainting and detection boundaries
    rgb[1530:1560, 1015:1040] = 0
    rgb[4180:4190, 100:120] = 0
    report = Pipeline(Settings(), Detector(), Painter()).clean(png(rgb), tmp_path, 'all-text')
    result = np.asarray(Image.open(tmp_path/'cleaned.png'))
    mask = np.asarray(Image.open(tmp_path/'mask.png')) > 0
    assert report['status'] == 'completed'
    assert result.shape == rgb.shape
    assert np.array_equal(result[~mask], rgb[~mask])
    assert (result[760:790, 740:790] == 255).all()
    assert (result[4180:4190, 100:120] == 255).all()


def test_empty_detection_is_review_not_success(tmp_path):
    report = Pipeline(Settings(), Detector(), Painter()).clean(png(np.full((100,100,3), 255, np.uint8)), tmp_path, 'all-text')
    assert report['status'] == 'needs_review'
    assert report['repaired_tiles'] == 0


def test_residual_text_requires_review(tmp_path):
    class Noop:
        def predict(self, rgb, mask): return rgb
    rgb = np.full((100,100,3), 255, np.uint8); rgb[30:60,30:60] = 0
    report = Pipeline(Settings(), Detector(), Noop()).clean(png(rgb), tmp_path, 'all-text')
    assert report['status'] == 'needs_review'
    assert report['residual_fraction'] == 1


def test_model_failure_never_writes_completed_result(tmp_path):
    class Broken:
        def predict(self, rgb, mask): raise RuntimeError('model failed')
    rgb = np.zeros((50,50,3), np.uint8)
    with pytest.raises(RuntimeError): Pipeline(Settings(), Detector(), Broken()).clean(png(rgb), tmp_path, 'all-text')
    assert not (tmp_path/'report.json').exists()


def test_dialogue_guard_excludes_open_background():
    im = Image.new('RGB', (300,300), 'white')
    draw = ImageDraw.Draw(im); draw.ellipse((50,50,250,250), outline='black', width=4)
    draw.rectangle((100,100,150,120), fill='black')
    rgb = np.asarray(im); text = np.zeros((300,300), bool); text[100:121,100:151] = True
    interior = bubble_interior(rgb, text)
    assert interior[110,120]
    assert not interior[10,10]
    assert not interior[50,150]


def test_pixels_rejected_before_decode():
    rgb = np.zeros((100,100,3), np.uint8)
    with pytest.raises(ValueError, match='pixel'): read_image(png(rgb), replace(Settings(), max_pixels=500))


def test_flat_bubble_removes_fine_strokes_without_neural_model(tmp_path):
    im = Image.new('RGB',(300,300),'white')
    draw=ImageDraw.Draw(im);draw.ellipse((40,40,260,260),outline='black',width=3)
    draw.rectangle((100,100,150,120),fill='black');draw.point((155,110),fill=(80,80,80))
    class GlyphDetector:
        calls=0
        def predict(self,rgb):
            mask=np.zeros(rgb.shape[:2],bool)
            if self.calls == 0: mask[100:121,100:151]=True
            self.calls+=1
            return [(90,90,80,50,.9)],mask
    class NoNeural:
        def predict(self,*args): raise AssertionError('Flat bubble should use its background colour')
    rgb=np.asarray(im)
    report=Pipeline(Settings(),GlyphDetector(),NoNeural()).clean(png(rgb),tmp_path,'dialogue')
    result=np.asarray(Image.open(tmp_path/'cleaned.png'))
    assert (result[100:121,100:151] == 255).all()
    assert (result[110,155] == 255).all()
    assert np.array_equal(result[40,150],rgb[40,150])
    assert report['flat_regions'] == 1


def test_smooth_colour_field_reconstructs_gradient_without_text_halo():
    from manga_cleaner.pipeline import smooth_background
    yy,xx=np.mgrid[:120,:180]
    expected=np.stack((220+xx*.05+yy*.1,225+xx*.05+yy*.08,235+xx*.05+yy*.05),axis=-1).round().astype(np.uint8)
    crop=expected.copy();crop[40:80,60:120]=0
    safe=np.ones((120,180),bool);safe[35:85,55:125]=False
    field=smooth_background(crop,safe)
    assert field is not None
    assert np.max(abs(field.astype(int)-expected.astype(int))) <= 1


def test_pale_text_outline_is_removed_even_if_segmentation_only_sees_dark_ink(tmp_path):
    im=Image.new('RGB',(300,300),(240,240,245))
    draw=ImageDraw.Draw(im);draw.ellipse((40,40,260,260),outline=(0,0,0),width=3)
    draw.rectangle((92,92,158,128),fill='white');draw.rectangle((100,100,150,120),fill='black')
    class InkOnly:
        calls=0
        def predict(self,rgb):
            mask=np.zeros(rgb.shape[:2],bool)
            if self.calls == 0: mask[100:121,100:151]=True
            self.calls+=1
            return [(90,90,80,50,.9)],mask
    class NoNeural:
        def predict(self,*args): raise AssertionError('Uniform bubble')
    report=Pipeline(Settings(),InkOnly(),NoNeural()).clean(png(np.asarray(im)),tmp_path,'dialogue')
    result=np.asarray(Image.open(tmp_path/'cleaned.png'))
    assert (result[92:129,92:159] == np.array([240,240,245])).all()


def test_ui_cluster_is_protected_and_duplicate_tiles_do_not_inflate_count():
    from manga_cleaner.pipeline import ui_protection
    boxes=[{'x':20,'y':6000+i*70,'width':180,'height':25} for i in range(4)]
    protected=ui_protection(boxes,14000,800)
    assert protected[6200,100]
    assert not protected[5500,100]
    assert not ui_protection([boxes[0]]*6,14000,800).any()
