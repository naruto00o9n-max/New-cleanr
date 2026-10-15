"""Balloon-aware quality pipeline with independent text and residual evidence."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .pipeline import Pipeline, read_image, bubble_interior, smooth_background, ui_protection
from .balloons import Balloon, merge_lines, neutral_clouds, punctuation_mask, contrasted_cloud, isolated_caption, display_punctuation, coloured_container


def polygon_support(line, shape, margin=0):
    h,w=shape;x,y,bw,bh=[line[k] for k in ('x','y','width','height')]
    result=np.zeros(shape,bool)
    result[max(0,y-margin):min(h,y+bh+margin),max(0,x-margin):min(w,x+bw+margin)]=True
    return result


def intersects(line, balloon):
    x,y,w,h=[line[k] for k in ('x','y','width','height')]
    bx,by,bw,bh=balloon.box
    left,top,right,bottom=max(x,bx),max(y,by),min(x+w,bx+bw),min(y+h,by+bh)
    if right<=left or bottom<=top:return 0
    coverage=float(balloon.mask[top-by:bottom-by,left-bx:right-bx].sum())/max(1,w*h)
    # Huge stylized words can touch a pointed balloon's sides. Their OCR
    # rectangle contains much blank space, so area coverage alone is misleading.
    cx,cy=x+w//2-bx,y+h//2-by
    if balloon.source.startswith('yolo') and 0<=cx<bw and 0<=cy<bh and balloon.mask[cy,cx] and coverage>.25:
        coverage=max(coverage,.60)
    return coverage


def refine_light_balloon(rgb, balloon):
    """Snap a learned polygon to an actual closed pale balloon interior.

    Network polygons can clip pointed corners; an enclosed background component
    restores these corners without treating surrounding page whitespace as ink.
    """
    x,y,w,h=balloon.box;crop=rgb[y:y+h,x:x+w]
    pixels=crop[balloon.mask]
    if len(pixels)<40 or np.percentile(pixels.mean(axis=1),70)<210:return balloon
    pad=max(64,int(max(w,h)*.2));left,top=max(0,x-pad),max(0,y-pad)
    right,bottom=min(rgb.shape[1],x+w+pad),min(rgb.shape[0],y+h+pad)
    roi=rgb[top:bottom,left:right]
    neutral=np.percentile(pixels.max(1).astype(int)-pixels.min(1).astype(int),95)<12
    # A grayscale gradient can be darker than 200 while still being inside the
    # same closed paper surface. Do not split its interior at that brightness.
    pale=(roi.min(axis=2)>(140 if neutral else 200)).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(pale,8)
    seed=np.zeros(pale.shape,bool);seed[y-top:y-top+h,x-left:x-left+w]=balloon.mask
    best=None;best_score=0
    for label in np.unique(labels[seed]):
        if label==0:continue
        cx,cy,cw,ch,area=stats[label]
        if cx==0 or cy==0 or cx+cw>=roi.shape[1] or cy+ch>=roi.shape[0] or area<80:continue
        component=labels==label;score=int((component&seed).sum())
        if score>best_score:best=component;best_score=score
    if best is None or best_score<seed.sum()*.20:return balloon
    contours,_=cv2.findContours(best.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    closed=np.zeros(pale.shape,np.uint8);cv2.drawContours(closed,contours,-1,1,-1)
    # A white hole inside a large outlined letter is not the balloon itself.
    # Refinement may expand corners, but must not clip the network hypothesis.
    if (closed.astype(bool)&seed).sum() < seed.sum()*.90:
        return balloon
    yy,xx=np.nonzero(closed);x0,y0,x1,y1=xx.min(),yy.min(),xx.max()+1,yy.max()+1
    return Balloon(left+int(x0),top+int(y0),closed[y0:y1,x0:x1].astype(bool),balloon.confidence,balloon.source+'-edge-refined')


def deduplicate_balloons(balloons):
    result=[]
    for balloon in sorted(balloons,key=lambda b:b.mask.sum(),reverse=True):
        x,y,w,h=balloon.box
        duplicate=False
        for other in result:
            ox,oy,ow,oh=other.box
            left,top,right,bottom=max(x,ox),max(y,oy),min(x+w,ox+ow),min(y+h,oy+oh)
            if right<=left or bottom<=top:continue
            intersection=int((balloon.mask[top-y:bottom-y,left-x:right-x]&other.mask[top-oy:bottom-oy,left-ox:right-ox]).sum())
            if intersection/max(1,min(int(balloon.mask.sum()),int(other.mask.sum())))>.72:
                duplicate=True;break
        if not duplicate:result.append(balloon)
    return result


def extend_neutral_text_support(rgb, balloon, boxes):
    """A model polygon must not cut a confirmed line in a neutral dark cloud."""
    x,y,w,h=balloon.box;pixels=rgb[y:y+h,x:x+w][balloon.mask]
    if not len(pixels) or np.median(pixels)>180:return balloon
    additions=[]
    for box in boxes:
        if box['confidence']<.65 or intersects(box,balloon)<.35:continue
        bx,by,bw,bh=[box[k] for k in ('x','y','width','height')]
        l,t,r,b=max(0,bx-12),max(0,by-12),min(rgb.shape[1],bx+bw+12),min(rgb.shape[0],by+bh+12)
        crop=rgb[t:b,l:r];chroma=crop.max(2).astype(int)-crop.min(2).astype(int)
        if np.percentile(chroma,98)>12 or np.median(crop)>185:continue
        additions.append((l,t,r,b))
    if not additions:return balloon
    l=min([x]+[a[0] for a in additions]);t=min([y]+[a[1] for a in additions])
    r=max([x+w]+[a[2] for a in additions]);b=max([y+h]+[a[3] for a in additions])
    mask=np.zeros((b-t,r-l),bool);mask[y-t:y-t+h,x-l:x-l+w]=balloon.mask
    for al,at,ar,ab in additions:mask[at-t:ab-t,al-l:ar-l]=True
    return Balloon(l,t,mask,balloon.confidence,balloon.source)


def refine_ink(crop, core, support, seed, field=None, colour=None):
    """Complete strokes using polarity and connected components, then dilate.

    A bright glyph on a smoky background uses bright ink seeds; its surrounding
    dark shadow is covered by dilation, not by masking the entire text rectangle.
    """
    pixels=crop[core&~support]
    if len(pixels)<20:pixels=crop[core]
    if not len(pixels):return np.zeros_like(core)
    background=np.median(pixels,axis=0) if colour is None else colour
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    if field is not None:
        difference=np.max(abs(crop.astype(float)-field.astype(float)),axis=2)
        candidate=difference>7
    elif np.max(np.std(pixels.astype(float),axis=0))<=5:
        difference=np.max(abs(crop.astype(float)-background),axis=2)
        candidate=difference>7
    elif background.mean()<130:
        threshold=max(95,float(background.mean())+35)
        candidate=gray>threshold
    else:
        candidate=gray<min(205,float(background.mean())-25)
    candidate &= support&core
    # Preserve full connected glyph components that are supported by the learned
    # mask or by a high-confidence line. Holes in letters are part of the glyph.
    count,labels,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8)
    refined=np.zeros(core.shape,np.uint8)
    for label in range(1,count):
        x,y,w,h,area=stats[label]
        if area<2:continue
        component=labels[y:y+h,x:x+w]==label
        if area>max(200,core.sum()*.92):continue
        refined[y:y+h,x:x+w][component]=1
    refined |= (seed&support&core).astype(np.uint8)
    contours,_=cv2.findContours(refined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(refined,contours,-1,1,-1)
    # Antialiasing, white outline and black shadow are all included.
    refined=cv2.dilate(refined,np.ones((7,7),np.uint8))>0
    return refined&core


def repair_boundary_fragments(result, original, balloon, lines, colour, inpainter):
    """Repair thick glyph fragments just outside an uncertain balloon polygon.

    Applies only to giant lettering on solid paper. Thin frame strokes are
    excluded by an opening operation; the tiny remaining patches use LaMa on
    the already-cleaned context, so a severed frame can be reconstructed.
    """
    x,y,w,h=balloon.box
    if not any(line['width']>w*.75 and line['height']>h*.55 for line in lines):return None
    pad=24;left,top=max(0,x-pad),max(0,y-pad);right,bottom=min(result.shape[1],x+w+pad),min(result.shape[0],y+h+pad)
    crop=result[top:bottom,left:right]
    predicted=np.zeros(crop.shape[:2],np.uint8)
    predicted[y-top:y-top+h,x-left:x-left+w]=balloon.mask.astype(np.uint8)
    inner=cv2.erode(predicted,np.ones((5,5),np.uint8))>0
    band=(cv2.dilate(predicted,np.ones((31,31),np.uint8))>0)&~inner
    support=np.zeros(band.shape,bool)
    for line in lines:
        lx,ly,lw,lh=[line[k] for k in ('x','y','width','height')]
        l,t,r,b=max(left,lx-10),max(top,ly-10),min(right,lx+lw+10),min(bottom,ly+lh+10)
        if r>l and b>t:support[t-top:b-top,l-left:r-left]=True
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    foreground=(gray<min(180,float(colour.mean())-50))&band&support
    rgb_chroma=crop.max(2).astype(int)-crop.min(2).astype(int)
    old=original[y:y+h,x:x+w]
    old_dark=old[balloon.mask&(cv2.cvtColor(old,cv2.COLOR_RGB2GRAY)<180)]
    if len(old_dark):
        ink_chroma=np.median(old_dark.max(1).astype(int)-old_dark.min(1).astype(int))
        if ink_chroma>12:foreground &= rgb_chroma>max(5,ink_chroma*.15)
    # Remove locally straight frame strokes in every direction. Small round
    # punctuation and glyph stubs remain, including a dot touching the frame.
    strokes=np.zeros(foreground.shape,np.uint8)
    for angle in range(0,180,15):
        radians=np.deg2rad(angle);dx=int(round(6*np.cos(radians)));dy=int(round(6*np.sin(radians)))
        kernel=np.zeros((13,13),np.uint8);cv2.line(kernel,(6-dx,6-dy),(6+dx,6+dy),1,1)
        strokes |= cv2.morphologyEx(foreground.astype(np.uint8),cv2.MORPH_OPEN,kernel)
    old_roi=original[top:bottom,left:right]
    dark=(cv2.cvtColor(old_roi,cv2.COLOR_RGB2GRAY)<min(180,float(colour.mean())-50))&support
    fat=cv2.distanceTransform(dark.astype(np.uint8),cv2.DIST_L2,5)>3
    if not fat.any():return None
    near=cv2.distanceTransform((~fat).astype(np.uint8),cv2.DIST_L2,5)<=16
    fragments=foreground&(strokes==0)&near
    count,labels,stats,_=cv2.connectedComponentsWithStats(fragments.astype(np.uint8),8)
    small=np.zeros_like(fragments)
    for label in range(1,count):
        fx,fy,fw,fh,area=stats[label]
        if 2<=area<=400 and fw<=60 and fh<=60:small|=labels==label
    repair=(cv2.dilate(small.astype(np.uint8),np.ones((5,5),np.uint8))>0)&band&support
    if not repair.any() or repair.sum()>balloon.mask.sum()*.02:return None
    yy,xx=np.nonzero(repair);padding=48
    l,t,r,b=max(0,xx.min()-padding),max(0,yy.min()-padding),min(crop.shape[1],xx.max()+padding+1),min(crop.shape[0],yy.max()+padding+1)
    generated=inpainter.predict(crop[t:b,l:r].copy(),repair[t:b,l:r])
    if generated.shape!=crop[t:b,l:r].shape or not np.isfinite(generated).all():raise RuntimeError('Invalid boundary repair result')
    crop[t:b,l:r][repair[t:b,l:r]]=generated[repair[t:b,l:r]]
    return left,top,repair


class QualityPipeline(Pipeline):
    def __init__(self,settings,detector,inpainter,segmenter,evidence):
        super().__init__(settings,detector,inpainter)
        self.segmenter,self.evidence=segmenter,evidence
        from .provenance import engine_fingerprint
        self.fingerprint=engine_fingerprint()

    def clean(self,data,output:Path,mode='dialogue',progress=lambda *args:None,manual_mask=None):
        if mode=='all-text':
            # Explicit all-text mode retains the pixel-mask neural path; it is
            # intentionally allowed to erase SFX and interface text as well.
            return super().clean(data,output,mode,progress,manual_mask)
        if mode!='dialogue':raise ValueError('Invalid cleaning mode')
        begun=time.monotonic();rgb=read_image(data,self.settings);h,w=rgb.shape[:2]
        proposals=self.segmenter.scan(rgb,progress)
        ocr_lines=self.evidence.scan(rgb,progress)
        ctd_mask,ctd_boxes=self.detect(rgb,progress)
        coarse_windows=[]
        for line in ocr_lines if h>1536 else []:
            if line['width']<w*.55 or line['height']<w*.18:continue
            if any(intersects(line,b)>.45 for b in proposals if b.confidence>=.60):continue
            cy=line['y']+line['height']//2;top=max(0,min(h-2048,cy-1024))
            if any(abs(top-old)<600 for old in coarse_windows):continue
            coarse_windows.append(top)
            crop=rgb[top:top+2048]
            regions,segmentation=self.detector.predict(crop)
            for bx,by,bw,bh,confidence in regions:
                if confidence<.50:continue
                ctd_boxes.append(dict(x=bx,y=top+by,width=bw,height=bh,confidence=round(confidence,4),scale='coarse'))
                ctd_mask[top+by:top+by+bh,bx:bx+bw] |= segmentation[by:by+bh,bx:bx+bw]
        learned=deduplicate_balloons([refine_light_balloon(rgb,b) for b in proposals if b.confidence>=.60])
        for candidate in deduplicate_balloons([b for b in proposals if b.confidence<.60]):
            x,y,bw,bh=candidate.box
            if any(intersects(dict(x=x,y=y,width=bw,height=bh),b)>.40 for b in learned):continue
            # The model can cut holes through giant punctuation. Detect its
            # disconnected stems/dots in the proposal rectangle, then restore
            # their full support before any background is estimated.
            punctuation=display_punctuation(rgb[y:y+bh,x:x+bw],np.ones_like(candidate.mask))
            supported=any(b['confidence']>=.65 and intersects(b,candidate)>.50 for b in ctd_boxes)
            readable=False
            if supported:
                reading=self.evidence.read_crop(rgb[y:y+bh,x:x+bw])
                readable=any(r['confidence']>=.75 and sum(c.isalpha() for c in r['text'])>=3 for r in reading)
            if readable or punctuation.any():
                if punctuation.any():candidate.mask |= cv2.dilate(punctuation.astype(np.uint8),np.ones((21,21),np.uint8))>0
                candidate.source='yolo11-confirmed-balloon'
                learned.append(candidate)
        lines=merge_lines([*ocr_lines,*[{**b,'confidence':b['confidence']*.7,'source':'comic-text-detector','text':''} for b in ctd_boxes]])
        clouds=neutral_clouds(rgb)
        balloons=[]
        for b in learned:
            if b.source=='yolo11-confirmed-balloon':
                bx,by,bw,bh=b.box
                colour_shape=coloured_container(rgb,dict(x=bx,y=by,width=bw,height=bh))
                if colour_shape is not None:b=colour_shape
            balloons.append(extend_neutral_text_support(rgb,b,ctd_boxes))
        balloons+=clouds
        protected=ui_protection(ctd_boxes,h,w)
        for box in ctd_boxes:
            if any(intersects(box,b)>.50 for b in balloons):continue
            x,y,bw,bh=[box[k] for k in ('x','y','width','height')]
            if protected[y:y+bh,x:x+bw].any():continue
            caption=isolated_caption(rgb,box)
            if caption is not None:balloons.append(caption)
            elif box['confidence']>=.45 and bw<w*.40 and bh<w*.15:
                # Small readable dialogue can straddle a balloon frame. A local
                # neural repair removes the letters and reconstructs the frame.
                reach=max(48,bh*2)
                nearby=any(intersects(dict(x=x-reach,y=y-reach,width=bw+reach*2,height=bh+reach*2),b)>.08 for b in learned)
                crop=rgb[y:y+bh,x:x+bw]
                if not nearby or not crop.size or (crop.min(2)>235).mean()<.72:continue
                chroma=crop.max(2).astype(int)-crop.min(2).astype(int)
                if np.percentile(chroma,95)>12:continue
                reading=self.evidence.read_crop(rgb[max(0,y-16):min(h,y+bh+16),max(0,x-16):min(w,x+bw+16)])
                if any(r['confidence']>=.85 and sum(c.isalpha() for c in r['text'])>=3 for r in reading):
                    l,t=max(0,x-12),max(0,y-12);r,b=min(w,x+bw+12),min(h,y+bh+12)
                    balloons.append(Balloon(l,t,np.ones((b-t,r-l),bool),float(box['confidence']),'edge-caption-repair'))
        # Recover closed narration boxes that the balloon segmenter can miss.
        # Only text-supported enclosures are admitted; interface clusters fail.
        for line in [*merge_lines(ctd_boxes),*lines]:
            if any(intersects(line,b)>.50 for b in balloons):continue
            px,py,pw,ph=[line[k] for k in ('x','y','width','height')]
            if protected[py:py+ph,px:px+pw].any():continue
            cloud=contrasted_cloud(rgb,line)
            if cloud is not None:
                balloons.append(cloud)
                continue
            x,y,bw,bh=[line[k] for k in ('x','y','width','height')]
            # Weak DB-only marks on devices/logos are not enough to invent a
            # narration balloon. Require independent comic-text evidence.
            supported=False;colour_only=True
            for box in ctd_boxes:
                bx,by,bw2,bh2=[box[k] for k in ('x','y','width','height')]
                overlap=max(0,min(x+bw,bx+bw2)-max(x,bx))*max(0,min(y+bh,by+bh2)-max(y,by))
                if box['confidence']>=.55 and overlap/max(1,min(bw*bh,bw2*bh2))>.50:
                    supported=True;colour_only=False;break
                if box.get('scale')=='coarse' and box['confidence']>=.50 and overlap/max(1,min(bw*bh,bw2*bh2))>.50:
                    supported=True
            if not supported:continue
            colour_cloud=coloured_container(rgb,line) or contrasted_cloud(rgb,line,allow_colour=True)
            if colour_cloud is not None:
                # Coloured scene art is not a caption merely because it is dark.
                # Require independently readable dialogue before this fallback.
                reading_line=max([line]+[r for r in ocr_lines if intersects(r,colour_cloud)>.45],key=lambda r:r['width']*r['height'])
                rx,ry,rw,rh=[reading_line[k] for k in ('x','y','width','height')]
                reading=self.evidence.read_crop(rgb[ry:ry+rh,rx:rx+rw])
                if not any(r['confidence']>=.65 and sum(c.isalpha() for c in r['text'])>=4 for r in reading):
                    reading=self.evidence.read_crop(rgb[max(0,ry-8):min(h,ry+rh+8),max(0,rx-8):min(w,rx+rw+8)])
                if any(r['confidence']>=.65 and sum(c.isalpha() for c in r['text'])>=4 for r in reading):
                    balloons.append(colour_cloud)
                    continue
            if colour_only:continue
            pad=max(96,bw//2,bh*2);left,top=max(0,x-pad),max(0,y-pad)
            right,bottom=min(w,x+bw+pad),min(h,y+bh+pad)
            crop=rgb[top:bottom,left:right]
            seed=ctd_mask[top:bottom,left:right].copy()
            # Independent OCR evidence supplements missing CTD pixels.
            support=np.zeros(seed.shape,bool)
            for other in lines:
                ox,oy,ow,oh=[other[k] for k in ('x','y','width','height')]
                l,t,r,b=max(left,ox),max(top,oy),min(right,ox+ow),min(bottom,oy+oh)
                if r>l and b>t:support[t-top:b-top,l-left:r-left]=True
            gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
            median=np.median(gray[~support]) if (~support).any() else 255
            seed |= support & (abs(gray.astype(float)-median)>25)
            interior=bubble_interior(crop,seed)
            if not interior.any():continue
            candidate=Balloon(left,top,interior,.5,'closed-narration-fallback')
            if intersects(line,candidate)>.55:balloons.append(candidate)
        balloons=deduplicate_balloons([complete_bright_glyphs(rgb,b,[*ocr_lines,*ctd_boxes]) for b in balloons])
        result=rgb.copy();mask=np.zeros((h,w),bool);selected=np.zeros_like(mask)
        accepted=[];audit=[];unresolved=[];flat=gradient=neural=punctuation=boundary=0
        protected_ui_blocks=[]
        for index,balloon in enumerate(balloons):
            x,y,bw,bh=balloon.box;crop=rgb[y:y+bh,x:x+bw]
            ui=protected[y:y+bh,x:x+bw]&balloon.mask
            # A learned detector can also mistake dense, tiny interface labels
            # for caption boxes. Independent cluster evidence vetoes these.
            if bh<w*.12 and ui.sum()>balloon.mask.sum()*.80:
                protected_ui_blocks.append(dict(x=x,y=y,width=bw,height=bh,source=balloon.source))
                continue
            # Polygon edges are uncertainty zones; visible bubble outlines stay.
            core=cv2.erode(balloon.mask.astype(np.uint8),np.ones((5,5),np.uint8),borderValue=0)>0
            related=[line for line in lines if intersects(line,balloon)>.48]
            # Text blocks and text lines are different evidence. A single OCR
            # line must not suppress a multiline CTD block (e.g. Arabic).
            ctd_related=[{**b,'source':'comic-text-detector','text':''} for b in ctd_boxes if intersects(b,balloon)>.48]
            related_for_mask=[*related,*ctd_related]
            support=np.zeros(core.shape,bool)
            margin=18 if balloon.source=='isolated-white-caption' else 10
            for line in related_for_mask:
                lx,ly,lw,lh=[line[k] for k in ('x','y','width','height')]
                left,top,right,bottom=max(x,lx-margin),max(y,ly-margin),min(x+bw,lx+lw+margin),min(y+bh,ly+lh+margin)
                if right>left and bottom>top:support[top-y:bottom-y,left-x:right-x]=True
            dots=np.zeros_like(core)
            if not related and balloon.source=='neutral-thought-container':
                dots=punctuation_mask(crop,core)
            if not related and (balloon.source.startswith('yolo') or balloon.source=='colour-enclosure'):
                dots=display_punctuation(crop,core,1 if balloon.confidence>=.60 else 2)
                if not dots.any():dots=punctuation_mask(crop,core)
            if dots.any():
                yy,xx=np.nonzero(dots);support[max(0,yy.min()-10):min(bh,yy.max()+11),max(0,xx.min()-10):min(bw,xx.max()+11)]=True
            if not related and not dots.any():continue
            safe=core&~cv2.dilate(support.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
            pixels=crop[safe]
            if len(pixels)<30:pixels=crop[core&~support]
            if len(pixels)<20:
                # Large display lettering can occupy the entire OCR rectangle.
                # Recover a known pale paper colour rather than silently skipping
                # the very balloon most in need of cleaning.
                pale=core&(crop.min(axis=2)>205)
                dark=core&(crop.max(axis=2)<180)
                edge=core&~cv2.erode(core.astype(np.uint8),np.ones((31,31),np.uint8),borderValue=0).astype(bool)
                paper_edge=edge.any() and (pale&edge).sum()>edge.sum()*.40
                if balloon.source.startswith('yolo') and paper_edge and pale.sum()>core.sum()*.15:
                    pixels=crop[pale]
                elif dark.sum()>core.sum()*.35:
                    # Giant white punctuation is ink, not pale paper. Retain
                    # the surrounding dark/coloured field as fitting evidence.
                    safe=dark
                    pixels=crop[dark]
                elif balloon.source.startswith('yolo') and pale.sum()>core.sum()*.15:
                    pixels=crop[pale]
                else:
                    pixels=crop[core&~ctd_mask[y:y+bh,x:x+bw]]
            if len(pixels)<20:
                unresolved.extend([{**r,'reason':'insufficient-background-evidence'} for r in related])
                continue
            colour=np.median(pixels,axis=0)
            is_flat=np.max(np.std(pixels.astype(float),axis=0))<=5
            if balloon.source=='edge-caption-repair':is_flat=False
            # Antialiased ink edges can contaminate a paper-colour sample even
            # though the balloon background itself is completely uniform.
            dominant=(np.max(abs(crop.astype(float)-colour),axis=2)<=3)&core
            if balloon.source.startswith('yolo') and colour.min()>205 and dominant.sum()>core.sum()*.45:
                is_flat=True
            colour_density=.60 if (support&core).sum()>core.sum()*.70 else .75 if balloon.source=='yolo11-confirmed-balloon' else .78
            colour_field=None if is_flat else coloured_field(crop,core,colour_density)
            row_field=None if is_flat else neutral_row_field(crop,safe)
            field=None if is_flat else colour_field if colour_field is not None else None if balloon.source=='colour-enclosure' else row_field if row_field is not None else smooth_background(crop,safe)
            # Classical enclosure recovery does not certify an opaque surface:
            # translucent captions can contain panel lines beneath the letters.
            # A smooth fit would erase these into a conspicuous blank rectangle.
            if balloon.source=='edge-caption-repair' or (balloon.source=='closed-narration-fallback' and not is_flat):
                field=None
            seed=ctd_mask[y:y+bh,x:x+bw]|dots
            ink=refine_ink(crop,core,support,seed,field,colour)
            if not ink.any():
                unresolved.extend([{**r,'reason':'empty-refined-ink'} for r in related])
                continue
            if row_field is not None and field is row_field:
                # Replace the full line, including low-contrast glyph interiors,
                # against independently sampled, per-row background evidence.
                ink=support&core
            # Complex containers need a continuous hole covering glyph interiors,
            # outlines and shadows; a stroke-shaped hole can leave letter-shaped
            # dark ghosts that recognition no longer notices.
            if not is_flat and field is None:
                if balloon.source=='edge-caption-repair' or (support&core).sum()<core.sum()*.70:
                    ink=support&core
            local=result[y:y+bh,x:x+bw]
            repair='solid-colour' if is_flat else 'smooth-field' if field is not None else 'anime-big-lama'
            if is_flat:
                local[ink]=np.round(colour).astype(np.uint8);flat+=1
            elif field is not None:
                local[ink]=field[ink];gradient+=1
            else:
                yy,xx=np.nonzero(ink);padding=64
                left,top=max(0,int(xx.min())-padding),max(0,int(yy.min())-padding)
                right,bottom=min(bw,int(xx.max())+padding+1),min(bh,int(yy.max())+padding+1)
                generated=self.inpainter.predict(crop[top:bottom,left:right],ink[top:bottom,left:right])
                if generated.shape!=crop[top:bottom,left:right].shape or not np.isfinite(generated).all():
                    raise RuntimeError('Invalid neural inpainting result')
                local[top:bottom,left:right][ink[top:bottom,left:right]]=generated[ink[top:bottom,left:right]];neural+=1
            mask[y:y+bh,x:x+bw] |= ink
            if is_flat and colour.min()>205 and balloon.source.startswith('yolo'):
                repaired=repair_boundary_fragments(result,rgb,balloon,related_for_mask,colour,self.inpainter)
                if repaired is not None:
                    left,top,extra=repaired
                    mask[top:top+extra.shape[0],left:left+extra.shape[1]] |= extra
                    boundary+=1
            selected[y:y+bh,x:x+bw] |= ctd_mask[y:y+bh,x:x+bw]&support&core
            accepted.extend(related_for_mask)
            if dots.any():punctuation+=1
            audit.append(dict(x=x,y=y,width=bw,height=bh,confidence=round(balloon.confidence,4),
                              source=balloon.source,repair=repair,ink_pixels=int(ink.sum()),texts=[r.get('text','') for r in related]))
            progress('cleaning-balloons',index+1,len(balloons))
        if manual_mask is not None:
            if manual_mask.shape!=(h,w):raise ValueError('Manual mask dimensions must match oriented image')
            binary=manual_mask>0;yy,xx=np.nonzero(binary)
            if len(xx):
                left,top,right,bottom=max(0,xx.min()-64),max(0,yy.min()-64),min(w,xx.max()+65),min(h,yy.max()+65)
                generated=self.inpainter.predict(rgb[top:bottom,left:right],binary[top:bottom,left:right])
                result[top:bottom,left:right][binary[top:bottom,left:right]]=generated[binary[top:bottom,left:right]];mask|=binary
        if not np.array_equal(result[~mask],rgb[~mask]):raise RuntimeError('Artwork preservation invariant failed')
        # Deduplicate within each detector while retaining its independent
        # coverage; OCR fragments must not hide a larger CTD paragraph.
        accepted=merge_lines([r for r in accepted if r.get('source')!='comic-text-detector'])+merge_lines([r for r in accepted if r.get('source')=='comic-text-detector'])
        residual=[];verification=[]
        for index,line in enumerate(accepted):
            x,y,bw,bh=[line[k] for k in ('x','y','width','height')]
            left,top,right,bottom=max(0,x-10),max(0,y-10),min(w,x+bw+10),min(h,y+bh+10)
            rows=self.evidence.read_crop(result[top:bottom,left:right])
            # Independent recognition, rather than reusing the CTD segmentation.
            significant=[row for row in rows if row['confidence']>=.65 and sum(c.isalpha() for c in row['text'])>=2]
            verification.append(dict(x=x,y=y,width=bw,height=bh,remaining=significant))
            residual.extend(significant)
            progress('verifying-text',index+1,len(accepted))
        rejected=[line for line in lines if not any(intersects(line,b)>.48 for b in balloons)]
        # Excluded SFX are expected in dialogue mode; doubtful or residual
        # dialogue evidence is exposed and never described as perfect accuracy.
        status='needs_review' if residual or unresolved or not audit else 'completed'
        report=dict(status=status,mode=mode,width=w,height=h,engine='balloon-aware-v2',
                    engine_fingerprint=self.fingerprint,
                    detected_pixels=int(ctd_mask.sum()),selected_pixels=int(selected.sum()),masked_pixels=int(mask.sum()),
                    excluded_pixels=int((ctd_mask&~selected).sum()),outside_mask_unchanged=True,
                    balloon_count=len(audit),learned_balloon_count=len(learned),thought_container_count=len(clouds),
                    flat_regions=flat,gradient_regions=gradient,repaired_tiles=neural,punctuation_regions=punctuation,boundary_repairs=boundary,
                    residual_text_count=len(residual),residual_text=residual,verification=verification,
                    elapsed_seconds=round(time.monotonic()-begun,2),boxes=lines,after_boxes=[],
                    excluded_regions=rejected,unresolved_regions=unresolved,protected_ui_blocks=protected_ui_blocks,balloons=audit)
        output.mkdir(parents=True,exist_ok=True)
        Image.fromarray(result).save(output/'cleaned.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(output/'mask.png')
        overlay=rgb.copy();overlay[mask]=(overlay[mask].astype(float)*.35+np.array([255,40,40])*.65).astype(np.uint8)
        Image.fromarray(overlay).save(output/'overlay.png')
        serialized=json.dumps(report,ensure_ascii=False,indent=2,default=lambda v: v.item() if isinstance(v,np.generic) else v.tolist())
        (output/'report.json').write_text(serialized)
        # CLI, SQLite jobs and chapter JSON all consume the same portable report.
        return json.loads(serialized)


def coloured_field(crop, core, minimum_density=.78):
    """Fit a coloured gradient/striped background without treating bright ink as paper.

    A robust row/column surface in log intensity preserves vertical texture as
    it fades into a dark gradient. Reject backgrounds it cannot explain.
    """
    if core.sum()<1000 or core.mean()<minimum_density:return None
    light=crop.min(2)>180
    # Near-black parts of the same bounded proposal are also background
    # evidence. A chromatic contour otherwise misses the black top of a banner.
    safe=(core|(crop.max(2)<45))&~cv2.dilate(light.astype(np.uint8),np.ones((31,31),np.uint8)).astype(bool)
    pixels=crop[safe].astype(float)
    if len(pixels)<core.sum()*.20:return None
    chroma=pixels.max(1)-pixels.min(1)
    # A dark cap may occupy most of a genuine coloured banner.
    # Require a substantial coloured portion rather than a coloured median.
    if np.percentile(chroma,75)<20 or np.median(pixels.mean(1))>155:return None
    useful=pixels[pixels.sum(1)>40]
    ratios=useful/np.maximum(useful.sum(1,keepdims=True),1)
    if not len(ratios):return None
    reference=np.median(ratios,axis=0)
    all_ratios=crop.astype(float)/np.maximum(crop.sum(2,keepdims=True).astype(float),1)
    # Pale page/glow at the edge of an imperfect polygon is not evidence for
    # the coloured background. Fit the dominant hue plus its near-black cap.
    safe &= (np.max(abs(all_ratios-reference),axis=2)<.15)|(crop.max(2)<45)
    if safe.sum()<core.sum()*.20:return None
    h,w=core.shape;values=np.log1p(crop.astype(float))
    rows=np.zeros((h,3));cols=np.zeros((w,3))
    for _ in range(6):
        for axis,profile,other in [(1,rows,cols),(0,cols,rows)]:
            for i in range(len(profile)):
                valid=safe[i] if axis==1 else safe[:,i]
                if valid.sum()<4:profile[i]=np.nan;continue
                sample=values[i] if axis==1 else values[:,i]
                profile[i]=np.median((sample-other)[valid],axis=0)
            valid=np.isfinite(profile).all(1)
            if valid.sum()<4:return None
            for channel in range(3):profile[:,channel]=np.interp(np.arange(len(profile)),np.flatnonzero(valid),profile[valid,channel])
    field=np.expm1(rows[:,None,:]+cols[None,:,:])
    error=np.max(abs(field-crop.astype(float)),axis=2)[safe]
    if np.percentile(error,85)>30 or np.percentile(error,95)>50:return None
    return np.clip(field,0,255).round().astype(np.uint8)


def neutral_row_field(crop,safe):
    """Robust row-wise affine sampling for neutral two-axis gradients."""
    if safe.sum()<100:return None
    pixels=crop[safe].astype(int)
    if np.percentile(pixels.max(1)-pixels.min(1),95)>8:return None
    h,w=safe.shape;profile=np.full((h,2,3),np.nan)
    for y in range(h):
        xs=np.flatnonzero(safe[y])
        if len(xs)<12 or np.ptp(xs)<max(12,w*.15):continue
        design=np.stack([np.ones(len(xs)),xs/max(1,w-1)],axis=1)
        values=crop[y,xs].astype(float);keep=np.ones(len(xs),bool)
        for _ in range(3):
            coefficients,_,rank,_=np.linalg.lstsq(design[keep],values[keep],rcond=None)
            if rank<2:break
            error=np.max(abs(design@coefficients-values),axis=1)
            keep=error<=max(2,float(np.percentile(error,85)))
        if rank==2:profile[y]=coefficients
    valid=np.isfinite(profile).all(axis=(1,2))
    if valid.sum()<max(10,h*.40):return None
    for coefficient in range(2):
        for channel in range(3):profile[:,coefficient,channel]=np.interp(np.arange(h),np.flatnonzero(valid),profile[valid,coefficient,channel])
    field=profile[:,0,None,:]+profile[:,1,None,:]*(np.arange(w)/max(1,w-1))[None,:,None]
    error=np.max(abs(crop.astype(float)-field),axis=2)[safe]
    if np.percentile(error,85)>3 or np.percentile(error,95)>7:return None
    return np.clip(field,0,255).round().astype(np.uint8).copy()


def complete_bright_glyphs(rgb,balloon,boxes):
    """Recover whole bright components clipped by a dark balloon prediction.

    Exterior white paper reaches the context boundary and is rejected. A glyph
    must intersect both the existing enclosure and independent text support.
    """
    x,y,w,h=balloon.box;pixels=rgb[y:y+h,x:x+w][balloon.mask]
    if not len(pixels) or np.median(pixels)>180:return balloon
    related=[box for box in boxes if intersects(box,balloon)>.30]
    if not related:return balloon
    pad=48;left,top=max(0,x-pad),max(0,y-pad)
    right,bottom=min(rgb.shape[1],x+w+pad),min(rgb.shape[0],y+h+pad)
    crop=rgb[top:bottom,left:right];shape=crop.shape[:2]
    original=np.zeros(shape,bool);original[y-top:y-top+h,x-left:x-left+w]=balloon.mask
    support=np.zeros(shape,bool)
    for box in related:
        bx,by,bw,bh=[box[k] for k in ('x','y','width','height')]
        l,t,r,b=max(left,bx-8),max(top,by-8),min(right,bx+bw+8),min(bottom,by+bh+8)
        if r>l and b>t:support[t-top:b-top,l-left:r-left]=True
    count,labels,stats,_=cv2.connectedComponentsWithStats((crop.min(2)>185).astype(np.uint8),8)
    additions=np.zeros(shape,np.uint8)
    for label in range(1,count):
        bx,by,bw,bh,area=stats[label]
        if area<5 or bx==0 or by==0 or bx+bw==shape[1] or by+bh==shape[0]:continue
        component=labels[by:by+bh,bx:bx+bw]==label
        if (component&support[by:by+bh,bx:bx+bw]).sum()<area*.80:continue
        overlap=int((component&original[by:by+bh,bx:bx+bw]).sum())
        if overlap==0 or overlap>=area:continue
        additions[by:by+bh,bx:bx+bw][component]=1
    if not additions.any():return balloon
    combined=original|(cv2.dilate(additions,np.ones((9,9),np.uint8))>0)
    yy,xx=np.nonzero(combined);l,t,r,b=int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
    return Balloon(left+l,top+t,combined[t:b,l:r],balloon.confidence,balloon.source)
