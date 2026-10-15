"""Full-page regression gate. Coordinates belong to evaluation, never processing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import httpx
import numpy as np
from PIL import Image,ImageDraw

from manga_cleaner.config import Settings
from manga_cleaner.pipeline import read_image
from manga_cleaner.runtime import create_pipeline
from manga_cleaner.provenance import engine_fingerprint


def checks(page, original, output):
    result=np.array(Image.open(output/'cleaned.png').convert('RGB'))
    mask=np.array(Image.open(output/'mask.png'))>0
    if result.shape!=original.shape or mask.shape!=original.shape[:2]:raise ValueError('Result dimensions changed')
    failures=[];targets=[]
    for target in page['targets']:
        x,y,w,h=target['box'];a=original[y:y+h,x:x+w];b=result[y:y+h,x:x+w]
        if x<0 or y<0 or min(w,h)<=0 or x+w>original.shape[1] or y+h>original.shape[0]:raise ValueError('Invalid annotation: '+target['name'])
        if target['kind'] not in {'paper','dark','coverage','reference_rows','channel_ceiling'}:raise ValueError('Unknown target metric')
        selection=np.ones((h,w),bool)
        if 'polygon' in target:
            selected_image=Image.new('1',(w,h));ImageDraw.Draw(selected_image).polygon([(px-x,py-y) for px,py in target['polygon']],fill=1)
            selection=np.array(selected_image,dtype=bool)
        if 'regions' in target:
            selection=np.zeros((h,w),bool)
            for rx,ry,rw,rh in target['regions']:
                selection[max(0,ry-y):min(h,ry-y+rh),max(0,rx-x):min(w,rx-x+rw)]=True
        if not selection.any():raise ValueError('Empty annotation selection')
        if target['kind'] in {'paper','dark','reference_rows','channel_ceiling'}:
            if target['kind']=='channel_ceiling':
                residual=float((b[:,:,target.get('channel',0)]>target['ceiling'])[selection].mean())
            elif target['kind']=='dark':
                residual=float((b.min(axis=2)>target.get('threshold',180))[selection].mean())
            elif target['kind']=='reference_rows':
                rx,ry,rw,rh=target['reference_box']
                if ry!=y or rh!=h or rw<=0:raise ValueError('Reference rows must align with the target')
                reference=np.median(original[ry:ry+rh,rx:rx+rw],axis=1)[:,None,:]
                if 'left_reference_box' in target:
                    lx,ly,lw,lh=target['left_reference_box']
                    if ly!=y or lh!=h or lw<=0:raise ValueError('Left reference rows must align')
                    left_reference=np.median(original[ly:ly+lh,lx:lx+lw],axis=1)[:,None,:]
                    position=(np.arange(x,x+w)-(lx+(lw-1)/2))/((rx+(rw-1)/2)-(lx+(lw-1)/2))
                    reference=left_reference+(reference-left_reference)*position[None,:,None]
                residual=float((np.max(abs(b.astype(float)-reference),axis=2)>target.get('tolerance',8))[selection].mean())
            else:
                residual=float((np.max(abs(b.astype(float)-np.array(target.get('colour',[255,255,255]))),axis=2)>target.get('tolerance',12))[selection].mean())
            limit=target.get('maximum_residual',.004)
            ok=residual<=limit
            measurement={'residual_fraction':round(residual,6)}
        else:
            background=np.array(target.get('background',np.median(a.reshape(-1,3),axis=0)))
            ink=(np.max(abs(a.astype(float)-background),axis=2)>target.get('contrast',35))&selection
            count=int(ink.sum())
            coverage=float(mask[y:y+h,x:x+w][ink].mean()) if count else 0
            ok=coverage>=target.get('minimum_coverage',.98)
            measurement={'ink_coverage':round(coverage,6),'source_contrast_pixels':count}
        targets.append({'name':target['name'],'kind':target['kind'],'passed':ok,**measurement})
        if not ok:failures.append(target['name'])
    protected=[]
    for region in page['protected']:
        x,y,w,h=region['box'];changed=int(np.any(result[y:y+h,x:x+w]!=original[y:y+h,x:x+w],axis=2).sum())
        protected.append({'name':region['name'],'changed_pixels':changed})
        if changed:failures.append('protected:'+region['name'])
    unchanged=np.array_equal(result[~mask],original[~mask])
    if not unchanged:failures.append('outside-mask')
    if not page['targets']:failures.append('missing-human-annotations')
    return {'passed':not failures,'failures':failures,'targets':targets,'protected':protected,'outside_mask_unchanged':unchanged}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('benchmarks/pages.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--download',action='store_true')
    p.add_argument('--split',choices=['all','regression','development','heldout'],default='all')
    p.add_argument('--check-only',action='store_true',help='Check already-rendered results; never labels them a new model run')
    p.add_argument('--cached-detector',action='store_true',help='Reuse versioned local input evidence; not an uncached throughput measurement')
    p.add_argument('--resume',action='store_true',help='Reuse complete outputs only when their processing fingerprint matches')
    p.add_argument('--compare',type=Path,help='Previous benchmark summary; reject any previously passing target regression')
    args=p.parse_args();settings=Settings.from_env();args.output.mkdir(parents=True,exist_ok=True)
    pages=json.loads(args.manifest.read_text())['pages'];pages=[r for r in pages if args.split=='all' or r['split']==args.split]
    if not pages:p.error('No selected benchmark pages')
    pipeline=None;rows=[]
    previous=json.loads(args.compare.read_text()) if args.compare else None
    for page in pages:
        path=Path(page['file'])
        if not path.exists() and args.download:
            response=httpx.get(page['url'],follow_redirects=True,timeout=90);response.raise_for_status()
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=page['sha256']:raise RuntimeError('Fixture changed: '+page['id'])
        target=args.output/page['id']
        report_path=target/'report.json'
        reusable=args.resume and report_path.exists() and all((target/name).exists() for name in ['cleaned.png','mask.png','overlay.png'])
        reusable=reusable and json.loads(report_path.read_text()).get('engine_fingerprint')==engine_fingerprint()
        if not args.check_only and not reusable:
            if pipeline is None:pipeline=create_pipeline(settings)
            if args.cached_detector:
                from manga_cleaner.evaluation import cache_input_detection
                cache_input_detection(pipeline,data)
            print('Processing',page['id'],flush=True)
            pipeline.clean(data,target,progress=lambda stage,done,total:print(stage,done,total,flush=True) if done==total else None)
        result=checks(page,read_image(data,settings),target)
        if previous:
            old=next((r for r in previous['pages'] if r['id']==page['id']),None)
            if old:
                passing={t['name'] for t in old['targets'] if t['passed']}
                old_targets={t['name']:t for t in old['targets']}
                for t in result['targets']:
                    if t['name'] in passing and not t['passed']:result['failures'].append('regression:'+t['name'])
                    before=old_targets.get(t['name'])
                    if before and before['passed']:
                        if before.get('kind')!=t['kind']:result['failures'].append('metric-changed:'+t['name'])
                        if 'residual_fraction' in before and t.get('residual_fraction',1)>before['residual_fraction']+.0001:
                            result['failures'].append('residual-increased:'+t['name'])
                        if 'ink_coverage' in before and t.get('ink_coverage',0)<before['ink_coverage']-.001:
                            result['failures'].append('coverage-decreased:'+t['name'])
                for name in passing-{t['name'] for t in result['targets']}:result['failures'].append('removed-target:'+name)
                result['passed']=not result['failures']
        report=json.loads((target/'report.json').read_text())
        if report.get('engine_fingerprint')!=engine_fingerprint():
            result['failures'].append('stale-engine-result');result['passed']=False
        row={'id':page['id'],'split':page['split'],'automatic_status':report['status'],**result};rows.append(row)
        print(page['id'],'PASS' if result['passed'] else 'FAIL',result['failures'],flush=True)
    fingerprint=engine_fingerprint()
    missing_pages=[]
    if previous and args.split=='all':missing_pages=sorted({r['id'] for r in previous['pages']}-{r['id'] for r in rows})
    summary={'schema':1,'engine_fingerprint':fingerprint,'check_only':args.check_only,'cached_detector':args.cached_detector,'passed':all(r['passed'] for r in rows) and not missing_pages,'missing_pages':missing_pages,'pages':rows,
             'scope':'Annotated text/background and protected-art regression checks; textured repairs still require visual review. Not a universal accuracy estimate.'}
    (args.output/'benchmark.json').write_text(json.dumps(summary,indent=2))
    return 0 if summary['passed'] else 1


if __name__=='__main__':sys.exit(main())
