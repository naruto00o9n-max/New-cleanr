"""Learned balloon polygons and independent PP-OCR text evidence.

Model provenance and inspiration are documented in docs/quality-research.md.
No page coordinates or particular phrases are part of the cleaning algorithm.
"""
import os
from dataclasses import dataclass
import cv2
import numpy as np
from .pipeline import starts


@dataclass
class Balloon:
    x: int
    y: int
    mask: np.ndarray
    confidence: float
    source: str

    def __post_init__(self):
        self.x,self.y=int(self.x),int(self.y)
        self.mask=np.asarray(self.mask,dtype=bool)
        self.confidence=float(self.confidence)

    @property
    def box(self):
        return self.x, self.y, self.mask.shape[1], self.mask.shape[0]


def merge_lines(lines):
    """Overlap-tile duplicates are evidence for one line, not two lines."""
    merged = []
    for line in sorted(lines, key=lambda b: b.get('confidence', 0), reverse=True):
        x, y, w, h = [line[k] for k in ('x','y','width','height')]
        duplicate = False
        for other in merged:
            ox, oy, ow, oh = [other[k] for k in ('x','y','width','height')]
            intersection = max(0,min(x+w,ox+ow)-max(x,ox))*max(0,min(y+h,oy+oh)-max(y,oy))
            if intersection / max(1,min(w*h,ow*oh)) > .65:
                duplicate = True; break
        if not duplicate: merged.append(line)
    return sorted(merged,key=lambda b:(b['y'],b['x']))


class BubbleSegmenter:
    def __init__(self, settings):
        os.environ.setdefault('YOLO_CONFIG_DIR', str(settings.data/'ultralytics'))
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(settings.threads)
        self.model = YOLO(str(settings.models/'speech-bubble-v11n.pt'))
        self.settings = settings

    def scan(self, rgb, progress=lambda *args: None):
        h,w=rgb.shape[:2]; size=1536; overlap=512
        windows=[(x,y) for y in starts(h,size,overlap) for x in starts(w,size,overlap)]
        balloons=[]
        for index,(left,top) in enumerate(windows):
            crop=rgb[top:top+size,left:left+size]
            # Keep weak candidates for independent confirmation; the processing
            # layer never treats this proposal threshold as an acceptance score.
            res=self.model.predict(np.ascontiguousarray(crop[:,:,::-1]),imgsz=1024,conf=.12,iou=.4,retina_masks=True,verbose=False,device='cpu')[0]
            if res.masks is not None:
                for box,mask,score in zip(res.boxes.xyxy.numpy(),res.masks.data.numpy(),res.boxes.conf.numpy()):
                    x0,y0,x1,y1=box.astype(int)
                    x0,y0=max(0,x0),max(0,y0);x1,y1=min(crop.shape[1],x1),min(crop.shape[0],y1)
                    local=(mask[y0:y1,x0:x1]>.5).astype(np.uint8)
                    # Text-shaped holes in a balloon prediction are not holes in
                    # the physical balloon. Retain its external polygon.
                    contours,_=cv2.findContours(local,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(local,contours,-1,1,-1)
                    if local.size and local.sum()>80:
                        balloons.append(Balloon(left+x0,top+y0,local.astype(bool),float(score),'yolo11-balloon'))
            progress('segmenting-balloons',index+1,len(windows))
        return balloons


class TextEvidence:
    def __init__(self, settings):
        from rapidocr_onnxruntime import RapidOCR
        self.ocr=RapidOCR(intra_op_num_threads=settings.threads,inter_op_num_threads=1)
        self.settings=settings

    def scan(self, rgb, progress=lambda *args: None):
        h,w=rgb.shape[:2];size=1024;overlap=256
        windows=[(x,y) for y in starts(h,size,overlap) for x in starts(w,size,overlap)]
        lines=[]
        for index,(left,top) in enumerate(windows):
            crop=np.ascontiguousarray(rgb[top:top+size,left:left+size,::-1])
            # Recognition confidence is not a detection confidence. Keeping only
            # readable words silently loses stylized GAAH and Arabic text.
            result,_=self.ocr(crop,use_cls=False,use_rec=False,box_thresh=.45)
            for quad in result or []:
                quad=np.asarray(quad);x0,y0=quad.min(axis=0);x1,y1=quad.max(axis=0)
                if x1-x0<5 or y1-y0<5:continue
                lines.append(dict(x=int(left+x0),y=int(top+y0),width=int(x1-x0+1),height=int(y1-y0+1),
                                  text='',confidence=.50,confidence_kind='minimum-detector-threshold',source='pp-ocr-detection'))
            progress('detecting-text',index+1,len(windows))
        return merge_lines(lines)

    def read_crop(self, rgb):
        result,_=self.ocr(np.ascontiguousarray(rgb[:,:,::-1]),use_cls=False,text_score=.40,box_thresh=.45)
        return [{'text':row[1],'confidence':float(row[2]),'polygon':np.asarray(row[0]).tolist()} for row in result or []]


def neutral_clouds(rgb):
    """Isolated grayscale thought containers on a black surrounding field.

    This complements the learned segmenter for smoky balloons it was not trained
    to identify. Colour artwork and components open to the image border fail.
    Text or a supported punctuation pattern is still required before painting.
    """
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    blurred=cv2.GaussianBlur(gray,(9,9),0)
    visible=(blurred>9).astype(np.uint8)
    contours,_=cv2.findContours(visible,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    balloons=[]
    h,w=gray.shape
    for contour in contours:
        x,y,bw,bh=cv2.boundingRect(contour)
        if x<=2 or y<=2 or x+bw>=w-2 or y+bh>=h-2 or bw*bh<5000 or bw*bh>800000 or bw/bh>5 or bh/bw>3:
            continue
        polygon=np.zeros((bh,bw),np.uint8)
        shifted=contour-np.array([[[x,y]]]);cv2.drawContours(polygon,[shifted],-1,1,-1)
        pixels=rgb[y:y+bh,x:x+bw][polygon>0]
        if len(pixels)<100:continue
        chroma=pixels.max(axis=1).astype(int)-pixels.min(axis=1).astype(int)
        if np.percentile(chroma,95)>12:continue
        # Context must be black on all sides, rather than skin/shadows in art.
        pad=12;left,top=max(0,x-pad),max(0,y-pad);right,bottom=min(w,x+bw+pad),min(h,y+bh+pad)
        context=gray[top:bottom,left:right]
        border=np.concatenate((context[0],context[-1],context[:,0],context[:,-1]))
        if np.percentile(border,95)>12:continue
        balloons.append(Balloon(x,y,polygon.astype(bool),.8,'neutral-thought-container'))
    return balloons


def punctuation_mask(rgb, container):
    """Find aligned, similarly sized ellipsis dots within an isolated cloud."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=gray.shape
    inner=cv2.erode(container.astype(np.uint8),np.ones((31,31),np.uint8),borderValue=0)>0
    yy,xx=np.mgrid[:h,:w]
    inner &= (abs(xx-w/2)<w*.30)&(abs(yy-h/2)<h*.25)
    if inner.sum()<50:return np.zeros_like(container)
    background=float(np.median(gray[inner]))
    if background>180:
        ink=gray<min(200,background-35)
    else:
        threshold=max(80,float(np.percentile(gray[inner],65))+25)
        ink=gray>threshold
    count,labels,stats,centroids=cv2.connectedComponentsWithStats((ink&inner).astype(np.uint8),8)
    dots=[]
    for index in range(1,count):
        x,y,bw,bh,area=stats[index]
        if 3<=area<=80 and 1<=bw<=12 and 1<=bh<=12 and .4<=bw/max(1,bh)<=2.5:
            dots.append(index)
    for anchor in dots:
        group=[index for index in dots if abs(centroids[index,1]-centroids[anchor,1])<5]
        if len(group)<3:continue
        group.sort(key=lambda i:centroids[i,0]);spacing=np.diff([centroids[i,0] for i in group])
        if spacing.size and 3<=np.median(spacing)<=24 and spacing.max()<np.median(spacing)*1.8:
            related=[i for i in dots if abs(centroids[i,1]-centroids[anchor,1])<30 and min(centroids[j,0] for j in group)-15<=centroids[i,0]<=max(centroids[j,0] for j in group)+15]
            return np.isin(labels,related)
    return np.zeros_like(container)


def display_punctuation(rgb, container, minimum_pairs=2):
    """Paired tall bright stems and baseline dots, e.g. a large ?!! balloon.

    Only used inside a learned enclosure proposal. This complements OCR, which
    often discards punctuation-only lines even when they are visually obvious.
    """
    h,w=container.shape
    if min(h,w)<80:return np.zeros_like(container)
    core=cv2.erode(container.astype(np.uint8),np.ones((9,9),np.uint8),borderValue=0)>0
    if not core.any():return np.zeros_like(container)
    pale=np.median(rgb[core])>180
    bright=((rgb.max(axis=2)<100) if pale else (rgb.min(axis=2)>215))&core
    if not .001<bright.sum()/core.sum()<.45:return np.zeros_like(container)
    count,labels,stats,centres=cv2.connectedComponentsWithStats(bright.astype(np.uint8),8)
    stems=[i for i in range(1,count) if stats[i,3]>max(15,h*(.12 if minimum_pairs==1 else .30)) and stats[i,2]<w*.38 and stats[i,4]>40]
    dots=[i for i in range(1,count) if h*.025<stats[i,3]<h*.22 and .35<stats[i,2]/stats[i,3]<2.8 and stats[i,4]>15]
    pairs=[]
    for stem in stems:
        x,y,bw,bh,area=stats[stem]
        for dot in dots:
            dx,dy,dw,dh,_=stats[dot]
            if y+bh<dy<y+bh+h*.20 and abs(centres[dot,0]-centres[stem,0])<max(bw*.65,w*.05):
                pairs.append((stem,dot));break
    if len(pairs)<minimum_pairs:return np.zeros_like(container)
    if np.ptp([centres[d,1] for _,d in pairs])>h*.12:return np.zeros_like(container)
    if minimum_pairs>1 and np.ptp([centres[s,0] for s,_ in pairs])<w*.12:return np.zeros_like(container)
    return np.isin(labels,[i for pair in pairs for i in pair])


def isolated_caption(rgb, line):
    """Multiline comic-text evidence surrounded by uninterrupted white paper.

    This deliberately rejects panel art, single SFX and device labels. It does
    not infer paper merely because the centre of a text box looks pale.
    """
    if line.get('confidence',0)<.65:return None
    x,y,bw,bh=[int(line[k]) for k in ('x','y','width','height')]
    pad=24;left,top=max(0,x-pad),max(0,y-pad)
    right,bottom=min(rgb.shape[1],x+bw+pad),min(rgb.shape[0],y+bh+pad)
    crop=rgb[top:bottom,left:right]
    if min(crop.shape[:2])<60:return None
    ring=np.ones(crop.shape[:2],bool);ring[12:-12,12:-12]=False
    if (crop[ring].min(axis=1)>245).mean()<.99:return None
    dark=crop.min(axis=2)<220
    if not .015<dark.mean()<.40:return None
    occupied=dark.sum(axis=1)>max(5,bw*.035)
    # Merge gaps from accents, but keep separate lines separate.
    occupied=cv2.morphologyEx(occupied.astype(np.uint8)[:,None],cv2.MORPH_CLOSE,np.ones((7,1),np.uint8))[:,0]>0
    edges=np.diff(np.pad(occupied.astype(int),(1,1)))
    heights=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    if sum(heights>=10)<2:return None
    return Balloon(left,top,np.ones(crop.shape[:2],bool),float(line['confidence']),'isolated-white-caption')


def contrasted_cloud(rgb, line, allow_colour=False):
    """Grayscale smoky container darker than a surrounding smooth gradient.

    Unlike neutral_clouds, this handles a black thought balloon over a light
    gradient. It requires an enclosed contrast component around existing text.
    """
    from .pipeline import smooth_background
    x,y,bw,bh=[line[k] for k in ('x','y','width','height')]
    pad=max(128,bw//2,bh*3)
    horizontal=max(pad,int(rgb.shape[1]*.65))
    left,top=max(0,x-horizontal),max(0,y-pad);right,bottom=min(rgb.shape[1],x+bw+horizontal),min(rgb.shape[0],y+bh+pad)
    crop=rgb[top:bottom,left:right];h,w=crop.shape[:2]
    if h<40 or w<40:return None
    chroma=crop.max(axis=2).astype(int)-crop.min(axis=2).astype(int)
    line_chroma=chroma[max(0,y-top):min(h,y-top+bh),max(0,x-left):min(w,x-left+bw)]
    if not line_chroma.size or (not allow_colour and np.percentile(line_chroma,95)>12):return None
    yy,xx=np.mgrid[:h,:w]
    cx,cy=x-left+bw/2,y-top+bh/2
    edge=max(12,w//25)
    lhs=np.median(crop[:,:edge].astype(float),axis=1)
    rhs=np.median(crop[:,-edge:].astype(float),axis=1)
    rows=(np.max(abs(lhs-rhs),axis=1)<15)&(lhs.max(1)-lhs.min(1)<10)&(rhs.max(1)-rhs.min(1)<10)
    row_indices=np.flatnonzero(rows)
    if len(row_indices)<h*.30:return None
    background=(lhs+rhs)/2
    field=np.stack([np.interp(np.arange(h),row_indices,background[row_indices,c]) for c in range(3)],axis=1)
    field=np.broadcast_to(field[:,None,:],crop.shape)
    difference=field.mean(axis=2)-crop.mean(axis=2)
    dark=cv2.GaussianBlur(difference.astype(np.float32),(9,9),0)>16
    if not allow_colour:dark &= chroma<12
    # A panel immediately below the thought container must not join the mask.
    invalid=np.flatnonzero(~rows & (np.arange(h)>cy+bh/2))
    if len(invalid):dark[int(invalid[0]):]=False
    invalid_above=np.flatnonzero(~rows & (np.arange(h)<y-top))
    if len(invalid_above):dark[:int(invalid_above[-1])+1]=False
    dark=dark.astype(np.uint8)
    contours,_=cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    options=[]
    for contour in contours:
        bx,by,sw,sh=cv2.boundingRect(contour)
        if bx<=1 or by<=1 or bx+sw>=w-1 or by+sh>=h-1 or sw<bw*1.10 or sh<bh*1.3:continue
        if not bx<cx<bx+sw or not by<cy<by+sh:continue
        polygon=np.zeros((h,w),np.uint8);cv2.drawContours(polygon,[contour],-1,1,-1)
        coverage=polygon[max(0,y-top):min(h,y-top+bh),max(0,x-left):min(w,x-left+bw)].sum()/max(1,bw*bh)
        if coverage<.60:continue
        options.append((sw*sh,bx,by,sw,sh,polygon))
    if not options:return None
    _,bx,by,sw,sh,polygon=min(options,key=lambda o:o[0])
    return Balloon(left+bx,top+by,polygon[by:by+sh,bx:bx+sw].astype(bool),.65,
                   'contrasted-colour-container' if allow_colour else 'contrasted-thought-container')


def coloured_container(rgb,line):
    """A coloured enclosure separated from panel art by its colour and paper.

    Used only with independent comic-text detection and recognized lettering.
    The contour may overlap a panel at its top; at least two other sides must
    have pale surrounding paper, preventing a whole coloured panel selection.
    """
    x,y,bw,bh=[int(line[k]) for k in ('x','y','width','height')]
    px=max(80,bw//2);py=max(80,bh//2)
    left,top=max(0,x-px),max(0,y-py);right,bottom=min(rgb.shape[1],x+bw+px),min(rgb.shape[0],y+bh+py)
    crop=rgb[top:bottom,left:right].astype(float);local=crop[y-top:y-top+bh,x-left:x-left+bw]
    chroma=local.max(2)-local.min(2)
    samples=local[(chroma>20)&(local.max(2)<210)]
    if len(samples)<bw*bh*.15:return None
    normalized=samples/np.maximum(samples.sum(1,keepdims=True),1)
    reference=np.median(normalized,axis=0)
    if reference.max()-reference.min()<.20:return None
    ratios=crop/np.maximum(crop.sum(2,keepdims=True),1)
    colour=(np.max(abs(ratios-reference),axis=2)<.13)&((crop.max(2)-crop.min(2))>8)
    colour=cv2.morphologyEx(colour.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    contours,_=cv2.findContours(colour,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    options=[]
    for contour in contours:
        bx,by,sw,sh=cv2.boundingRect(contour)
        if sw<bw*.85 or sh<bh*.85 or sw*sh>bw*bh*5:continue
        polygon=np.zeros(colour.shape,np.uint8);cv2.drawContours(polygon,[contour],-1,1,-1)
        coverage=polygon[y-top:y-top+bh,x-left:x-left+bw].mean()
        if coverage<.55:continue
        al,at,ar,ab=left+bx,top+by,left+bx+sw,top+by+sh
        edges=[]
        for strip in [rgb[max(0,at-16):at,al:ar],rgb[ab:min(rgb.shape[0],ab+16),al:ar],
                      rgb[at:ab,max(0,al-16):al],rgb[at:ab,ar:min(rgb.shape[1],ar+16)]]:
            if strip.size:edges.append(float((strip.min(2)>180).mean()))
        if sum(v>.60 for v in edges)<2 or not any(v>.90 for v in edges):continue
        options.append((sw*sh,bx,by,sw,sh,polygon))
    if not options:return None
    _,bx,by,sw,sh,polygon=min(options,key=lambda v:v[0])
    return Balloon(left+bx,top+by,polygon[by:by+sh,bx:bx+sw].astype(bool),.65,'colour-enclosure')
