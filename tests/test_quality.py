"""Meaningful quality regressions: coverage, outlines, UI safety and failed repair."""
from dataclasses import replace
import io
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from manga_cleaner.config import Settings
from manga_cleaner.balloons import Balloon, neutral_clouds, merge_lines, punctuation_mask, contrasted_cloud
from manga_cleaner.quality import QualityPipeline, refine_light_balloon


def png(image):
    buffer=io.BytesIO();image.save(buffer,format='PNG');return buffer.getvalue()

class NoCTD:
    def predict(self,rgb):return [],np.zeros(rgb.shape[:2],bool)

class FixedBalloons:
    def __init__(self,balloons):self.balloons=balloons
    def scan(self,rgb,progress):return self.balloons

class Evidence:
    def __init__(self,lines,residual=None):self.lines=lines;self.residual=residual or []
    def scan(self,rgb,progress):return self.lines
    def read_crop(self,rgb):return self.residual

class NeverNeural:
    def predict(self,*args):raise AssertionError('Solid white paper should not need LaMa')


def test_giant_stylized_word_is_cleaned_without_recognition_or_ctd(tmp_path):
    im=Image.new('RGB',(400,260),'white');draw=ImageDraw.Draw(im)
    draw.rectangle((45,40,355,220),outline='black',width=3)
    draw.text((70,75),'GAAH!',font=ImageFont.load_default(size=90),fill=(20,20,80))
    mask=np.ones((174,304),bool)
    # OCR rectangle fills the balloon, so there are no outside-box samples.
    line=dict(x=44,y=39,width=313,height=183,confidence=.5,text='',source='pp-ocr-detection')
    p=QualityPipeline(Settings(),NoCTD(),NeverNeural(),FixedBalloons([Balloon(48,43,mask,.95,'yolo11-balloon')]),Evidence([line]))
    report=p.clean(png(im),tmp_path)
    actual=np.array(Image.open(tmp_path/'cleaned.png'))
    assert (actual[65:175,65:330] == 255).all()
    assert np.array_equal(actual[40],np.array(im)[40])
    assert report['flat_regions']==1
    assert report['status']=='completed'
    json.dumps(report)  # persistent job and Drive report contract


def test_ctd_paragraph_is_not_lost_when_ocr_only_finds_one_fragment(tmp_path):
    im=Image.new('RGB',(360,280),'white');d=ImageDraw.Draw(im)
    d.ellipse((30,30,330,250),outline='black',width=3)
    d.text((80,80),'FIRST',font=ImageFont.load_default(size=35),fill='black')
    d.text((80,140),'SECOND',font=ImageFont.load_default(size=35),fill='black')
    source=np.array(im)
    class ParagraphCTD:
        def predict(self,rgb):
            text=(rgb[:,:,0]<50)&(np.indices(rgb.shape[:2])[0]>70)&(np.indices(rgb.shape[:2])[0]<190)
            return [(65,70,225,120,.95)],text
    line=dict(x=80,y=75,width=130,height=45,confidence=.5,text='',source='pp-ocr-detection')
    core=np.zeros((230,320),np.uint8);cv2.ellipse(core,(160,110),(147,106),0,0,360,1,-1)
    p=QualityPipeline(Settings(),ParagraphCTD(),NeverNeural(),FixedBalloons([Balloon(20,20,core.astype(bool),.95,'yolo11-balloon')]),Evidence([line]))
    report=p.clean(png(im),tmp_path)
    actual=np.array(Image.open(tmp_path/'cleaned.png'))
    assert (actual[75:180,65:295] == 255).all()
    final_mask=np.array(Image.open(tmp_path/'mask.png'))>0
    assert np.array_equal(actual[~final_mask],source[~final_mask])


def test_residual_recognition_marks_needs_review(tmp_path):
    im=Image.new('RGB',(250,200),'white');d=ImageDraw.Draw(im)
    d.ellipse((20,20,230,180),outline='black',width=3);d.text((70,80),'TEXT',font=ImageFont.load_default(size=30),fill='black')
    line=dict(x=65,y=75,width=100,height=45,confidence=.5,source='pp-ocr-detection')
    mask=np.zeros((200,250),np.uint8);cv2.ellipse(mask,(125,100),(100,75),0,0,360,1,-1)
    remaining=[{'text':'TEXT','confidence':.9,'polygon':[[0,0],[10,0],[10,10],[0,10]]}]
    p=QualityPipeline(Settings(),NoCTD(),NeverNeural(),FixedBalloons([Balloon(0,0,mask.astype(bool),.9,'yolo11-balloon')]),Evidence([line],remaining))
    report=p.clean(png(im),tmp_path)
    assert report['status']=='needs_review'
    assert report['residual_text_count']==1


def test_coloured_art_with_text_is_not_a_grayscale_thought_container():
    im=Image.new('RGB',(400,300),(180,40,20));d=ImageDraw.Draw(im)
    d.ellipse((70,60,330,240),fill=(20,80,170));d.text((130,120),'BANG',font=ImageFont.load_default(size=32),fill='white')
    line=dict(x=120,y=110,width=150,height=55)
    rgb=np.array(im)
    assert neutral_clouds(rgb)==[]
    assert contrasted_cloud(rgb,line) is None


def test_ellipsis_dots_are_completed_without_an_ocr_word():
    rgb=np.zeros((240,400,3),np.uint8);cv2.ellipse(rgb,(200,120),(140,75),0,0,360,(45,45,45),-1)
    for x in range(158,230,12):cv2.circle(rgb,(x,120),2,(140,140,140),-1)
    balloons=neutral_clouds(rgb)
    assert len(balloons)==1
    balloon=balloons[0];x,y,w,h=balloon.box
    dots=punctuation_mask(rgb[y:y+h,x:x+w],balloon.mask)
    assert dots.sum()>40


def test_white_glyph_hole_cannot_replace_the_entire_balloon_polygon():
    im=Image.new('RGB',(320,240),'white');d=ImageDraw.Draw(im)
    d.rectangle((30,25,290,215),outline='black',width=3)
    d.rectangle((45,35,280,210),fill=(25,25,80));d.rectangle((120,90,200,170),fill='white')
    original=Balloon(33,28,np.ones((185,255),bool),.95,'yolo11-balloon')
    refined=refine_light_balloon(np.array(im),original)
    assert refined.mask.size>=original.mask.size*.9


def test_overlap_deduplication_keeps_disjoint_text_lines():
    a=dict(x=20,y=50,width=100,height=30,confidence=.5)
    b=dict(x=21,y=51,width=101,height=29,confidence=.6)
    c=dict(x=20,y=90,width=100,height=30,confidence=.5)
    assert len(merge_lines([a,b,c]))==2


def test_interface_cluster_vetoes_a_false_learned_caption(tmp_path):
    im=Image.new('RGB',(600,600),(180,205,220));draw=ImageDraw.Draw(im)
    boxes=[]
    for index in range(4):
        y=150+index*65
        draw.text((150,y),'INTERFACE',font=ImageFont.load_default(size=20),fill='black')
        boxes.append(dict(x=145,y=y-2,width=130,height=26,confidence=.9))
    class InterfaceCTD:
        def predict(self,rgb):
            return [(b['x'],b['y'],b['width'],b['height'],b['confidence']) for b in boxes],rgb[:,:,0]<50
    predicted=Balloon(140,140,np.ones((40,150),bool),.95,'yolo11-balloon')
    line={**boxes[0],'source':'pp-ocr-detection'}
    p=QualityPipeline(Settings(),InterfaceCTD(),NeverNeural(),FixedBalloons([predicted]),Evidence([line]))
    report=p.clean(png(im),tmp_path)
    assert report['protected_ui_blocks']
    assert np.array_equal(np.array(Image.open(tmp_path/'cleaned.png')),np.array(im))


def test_border_repair_never_touches_a_distant_frame_corner():
    from manga_cleaner.quality import repair_boundary_fragments
    im=Image.new('RGB',(450,320),'white');d=ImageDraw.Draw(im)
    d.rectangle((50,45,400,275),outline=(20,20,90),width=3)
    d.text((70,90),'GAAH',font=ImageFont.load_default(size=95),fill=(20,20,90))
    source=np.array(im);result=source.copy()
    result[80:210,65:385]=255
    # Small glyph fragment immediately near the original thick lettering.
    result[112:132,62:70]=(20,20,90)
    polygon=np.ones((224,344),bool)
    balloon=Balloon(53,48,polygon,.95,'yolo11-balloon')
    line=dict(x=50,y=40,width=353,height=238)
    class WhitePainter:
        def predict(self,rgb,mask):return np.full_like(rgb,255)
    repaired=repair_boundary_fragments(result,source,balloon,[line],np.array([255,255,255]),WhitePainter())
    assert np.array_equal(result[45,397],source[45,397])
    if repaired is not None:
        left,top,mask=repaired
        assert int(mask.sum())<int(polygon.sum()*.02)


def test_model_numpy_coordinates_are_normalized_before_job_serialization():
    balloon=Balloon(np.int64(10),np.int64(20),np.ones((30,40),np.uint8),np.float32(.9),'yolo11-balloon')
    assert isinstance(balloon.x,int)
    assert isinstance(balloon.y,int)
    assert isinstance(balloon.confidence,float)
    json.dumps({'box':balloon.box,'confidence':balloon.confidence})


def test_isolated_multiline_caption_requires_clean_paper_surrounding():
    from manga_cleaner.balloons import isolated_caption
    im=Image.new('RGB',(420,300),'white');d=ImageDraw.Draw(im)
    font=ImageFont.load_default(size=32)
    d.text((100,90),'FIRST LINE',font=font,fill=(100,70,40))
    d.text((100,145),'SECOND LINE',font=font,fill=(100,70,40))
    line=dict(x=95,y=90,width=225,height=90,confidence=.9)
    assert isolated_caption(np.array(im),line) is not None
    # The same text in artwork must not imply an empty paper rectangle.
    d.rectangle((70,65,86,205),fill=(100,160,200))
    assert isolated_caption(np.array(im),line) is None
    assert isolated_caption(np.array(im),{**line,'confidence':.4}) is None


def test_large_punctuation_recovers_stems_and_dots_outside_a_model_hole():
    from manga_cleaner.balloons import display_punctuation
    im=Image.new('RGB',(480,300),(65,5,10));d=ImageDraw.Draw(im)
    for x in [140,250,350]:
        d.rectangle((x,50,x+20,195),fill='white')
        d.ellipse((x,220,x+22,244),fill='white')
    ink=display_punctuation(np.array(im),np.ones((300,480),bool))
    assert ink[100,150] and ink[100,260] and ink[100,360]
    assert ink[232,150] and not ink[20,20]


def test_single_exclamation_on_paper_does_not_need_word_recognition(tmp_path):
    im=Image.new('RGB',(320,240),'white');d=ImageDraw.Draw(im)
    d.ellipse((40,30,280,210),outline='black',width=3)
    d.rectangle((153,80,165,128),fill='black');d.ellipse((153,140,165,152),fill='black')
    mask=np.zeros((240,320),np.uint8);cv2.ellipse(mask,(160,120),(116,86),0,0,360,1,-1)
    pipeline=QualityPipeline(Settings(),NoCTD(),NeverNeural(),FixedBalloons([Balloon(0,0,mask,.95,'yolo11-balloon')]),Evidence([]))
    report=pipeline.clean(png(im),tmp_path)
    actual=np.array(Image.open(tmp_path/'cleaned.png'))
    assert (actual[75:160,145:175]==255).all()
    assert report['punctuation_regions']==1


def test_neutral_cloud_polygon_cannot_clip_a_confirmed_line():
    from manga_cleaner.quality import extend_neutral_text_support
    rgb=np.full((300,400,3),70,np.uint8)
    balloon=Balloon(60,100,np.ones((150,280),bool),.95,'yolo11-balloon')
    line=dict(x=110,y=85,width=160,height=80,confidence=.9)
    expanded=extend_neutral_text_support(rgb,balloon,[line])
    assert expanded.y<=line['y']-10
    # Coloured illustration adjacent to a prediction is not cloud background.
    rgb[:100]=(70,110,150)
    unchanged=extend_neutral_text_support(rgb,balloon,[line])
    assert unchanged.box==balloon.box


def test_coloured_nonlinear_gradient_is_recovered_without_white_paper_hallucination(tmp_path):
    h,w=360,540;rgb=np.full((h,w,3),255,np.uint8)
    yy=np.arange(280)/279
    colour=np.stack([25+100*yy**3,3+10*yy**3,7+20*yy**3],axis=1).round().astype(np.uint8)
    rgb[40:320,50:490]=colour[:,None,:]
    original=rgb.copy()
    for x in [170,280,390]:
        rgb[80:225,x:x+16]=255
        cv2.circle(rgb,(x+8,255),10,(255,255,255),-1)
    proposed=np.ones((280,440),bool)
    # The learned polygon cuts through letters; proof must use the full proposal.
    proposed[:,230:250]=False
    pipeline=QualityPipeline(Settings(),NoCTD(),NeverNeural(),FixedBalloons([Balloon(50,40,proposed,.25,'yolo11-balloon')]),Evidence([]))
    report=pipeline.clean(png(Image.fromarray(rgb)),tmp_path)
    result=np.array(Image.open(tmp_path/'cleaned.png'))
    assert report['balloon_count']==1
    assert np.max(abs(result[80:275,150:420].astype(int)-original[80:275,150:420]))<=5
    assert np.array_equal(result[:35],original[:35])


def test_overlapping_rectangles_do_not_discard_a_distinct_frame_caption():
    from manga_cleaner.quality import deduplicate_balloons
    large=np.zeros((400,500),bool);large[:,200:]=True;large[250:,:]=True
    portrait_balloon=Balloon(0,0,large,.9,'yolo11-balloon')
    # A small line near the frame lies inside the large bounding rectangle,
    # but most of its actual pixels are outside the learned polygon.
    caption=Balloon(150,80,np.ones((60,100),bool),.9,'edge-caption-repair')
    assert len(deduplicate_balloons([portrait_balloon,caption]))==2
    duplicate=Balloon(0,0,large.copy(),.8,'yolo11-balloon')
    assert len(deduplicate_balloons([portrait_balloon,duplicate]))==1


def test_dark_giant_ink_is_not_mistaken_for_a_dark_balloon_background(tmp_path):
    im=Image.new('RGB',(400,300),'white');d=ImageDraw.Draw(im)
    d.rectangle((30,25,370,275),outline='black',width=3)
    # Large dark lettering occupies >35% of the interior, but the enclosure
    # boundary still provides direct evidence of white paper.
    d.rectangle((75,70,325,230),fill=(20,20,90))
    line=dict(x=33,y=28,width=334,height=244,confidence=.5,source='pp-ocr-detection')
    b=Balloon(34,29,np.ones((242,332),bool),.95,'yolo11-balloon')
    p=QualityPipeline(Settings(),NoCTD(),NeverNeural(),FixedBalloons([b]),Evidence([line]))
    p.clean(png(im),tmp_path)
    actual=np.array(Image.open(tmp_path/'cleaned.png'))
    assert (actual[60:240,65:335]==255).all()


def test_neutral_gradient_has_no_letter_shaped_fitting_residue(tmp_path):
    yy=np.arange(300)/299
    profile=(240-90*yy**3).round().astype(np.uint8)
    expected=np.clip(profile[:,None,None].astype(float)+np.arange(400)[None,:,None]*.035,0,255)
    expected=np.broadcast_to(expected,(300,400,3)).round().astype(np.uint8).copy()
    im=Image.fromarray(expected);d=ImageDraw.Draw(im)
    d.rectangle((25,25,375,275),outline='black',width=3)
    d.text((80,95),'WHAT',font=ImageFont.load_default(size=46),fill=(90,90,90))
    d.text((65,150),'HAPPENED?',font=ImageFont.load_default(size=40),fill=(100,100,100))
    line=dict(x=60,y=90,width=270,height=110,confidence=.5,source='pp-ocr-detection')
    b=Balloon(29,29,np.ones((242,342),bool),.95,'yolo11-balloon')
    p=QualityPipeline(Settings(),NoCTD(),NeverNeural(),FixedBalloons([b]),Evidence([line]))
    p.clean(png(im),tmp_path)
    actual=np.array(Image.open(tmp_path/'cleaned.png'))
    assert np.max(abs(actual[90:200,60:330].astype(int)-expected[90:200,60:330]))<=2


def test_bright_letter_head_outside_a_prediction_is_completed(tmp_path):
    im=Image.new('RGB',(400,320),(20,20,20));d=ImageDraw.Draw(im)
    d.rectangle((120,80,145,190),fill='white')
    d.rectangle((190,85,215,190),fill='white')
    line=dict(x=110,y=75,width=120,height=125,confidence=.5,source='pp-ocr-detection')
    # Predicted dark enclosure starts below both letter heads.
    b=Balloon(75,110,np.ones((155,230),bool),.95,'yolo11-balloon')
    p=QualityPipeline(Settings(),NoCTD(),NeverNeural(),FixedBalloons([b]),Evidence([line]))
    p.clean(png(im),tmp_path)
    actual=np.array(Image.open(tmp_path/'cleaned.png'))
    assert (actual[80:195,115:220]==20).all()


def test_coloured_surface_with_a_majority_black_cap_is_not_rejected():
    from manga_cleaner.quality import coloured_field
    h,w=320,500
    y=np.arange(h)/h
    expected=np.stack([5+175*y**4,1+7*y**4,2+10*y**4],axis=1)
    expected=np.broadcast_to(expected[:,None,:],(h,w,3)).round().astype(np.uint8).copy()
    observed=expected.copy()
    observed[50:230,160:190]=255
    observed[60:250,290:320]=255
    core=np.ones((h,w),bool)
    field=coloured_field(observed,core)
    assert field is not None
    assert np.max(abs(field[50:250,160:320].astype(int)-expected[50:250,160:320]))<=3


def test_quality_gate_detects_coloured_ghosts_after_white_letters_are_gone(tmp_path):
    import runpy
    from pathlib import Path
    checks=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/quality_benchmark.py"))["checks"]
    original=np.full((20,30,3),255,np.uint8)
    page={'targets':[dict(name='dark cap',kind='channel_ceiling',box=[0,0,30,20],channel=0,ceiling=45)],'protected':[]}
    Image.fromarray(np.full((20,30),255,np.uint8)).save(tmp_path/'mask.png')
    output=np.zeros_like(original);output[4:12,8:14,0]=140
    Image.fromarray(output).save(tmp_path/'cleaned.png')
    assert not checks(page,original,tmp_path)['passed']
    output[:]=0;Image.fromarray(output).save(tmp_path/'cleaned.png')
    assert checks(page,original,tmp_path)['passed']
