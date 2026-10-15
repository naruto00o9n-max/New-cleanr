import json
from pathlib import Path
import shutil
import zipfile


def bundle_results(output, delivery):
    """Keep finished pages separate from diagnostic images in the Drive folder."""
    delivery.mkdir(parents=True, exist_ok=True)
    pages = sorted(output.glob('*/cleaned.png'))
    if not pages: raise RuntimeError('No completed pages to deliver')
    for page in pages:
        shutil.copyfile(page, delivery / (page.parent.name + '.png'))
    summary = output / 'chapter-report.json'
    if summary.exists(): shutil.copyfile(summary, delivery / summary.name)
    with zipfile.ZipFile(delivery / 'review-artifacts.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name != 'cleaned.png':
                archive.write(path, path.relative_to(output))
    return delivery


def direct_results(root, manifest, source):
    """Mirror original delivery: attachments -> images, ZIP -> ZIP, Drive -> Drive."""
    if source.get('kind') not in {'images','zip'}:return False
    delivery=root/'discord-delivery';shutil.rmtree(delivery,ignore_errors=True);delivery.mkdir()
    import re
    used=set()
    for index,item in enumerate(manifest,1):
        stem=re.sub(r'[\\/\r\n\x00-\x1f]','_',Path(item['source_name']).stem)[:150] or str(index)
        name=stem+'-clean.png'
        suffix=index
        while name.casefold() in used:
            name=f'{stem}-{suffix}-clean.png';suffix+=1
        used.add(name.casefold());shutil.copyfile(root/'output'/f'{index:04d}'/'cleaned.png',delivery/name)
    if source['kind']=='zip':
        name=re.sub(r'[\\/\r\n\x00-\x1f]','_',source.get('name','chapter'))[:150] or 'chapter'
        archive=root/(name+'-cleaned.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
            for path in sorted(delivery.iterdir()):z.write(path,path.name)
        shutil.rmtree(delivery);delivery.mkdir();shutil.move(archive,delivery/archive.name)
    if sum(p.stat().st_size for p in delivery.iterdir())>int(source.get('discord_limit',24*1024*1024)):
        shutil.rmtree(delivery);return False
    return True

class Worker:
    def __init__(self, settings, store, pipeline, drive):
        self.settings, self.store, self.pipeline, self.drive = settings, store, pipeline, drive

    def once(self):
        job = self.store.claim()
        if not job: return None
        identifier = job['id']
        root = self.settings.data / identifier
        output = root / 'output'
        phase = 'processing'
        try:
            manifest = json.loads((root / 'input' / 'manifest.json').read_text())
            reports = []
            for index, item in enumerate(manifest, 1):
                target = output / f'{index:04d}'
                report_file = target / 'report.json'
                # A complete page is reusable after restart or upload failure.
                if report_file.exists():
                    report = json.loads(report_file.read_text())
                else:
                    def progress(stage, done, total):
                        self.store.patch(identifier, progress=f'{index}/{len(manifest)} {stage} {done}/{total}')
                    report = self.pipeline.clean((root / 'input' / item['path']).read_bytes(), target, job['mode'], progress)
                reports.append({'page': index, 'source_name': item['source_name'], **report})
            review = any(report['status'] != 'completed' for report in reports)
            output.mkdir(parents=True, exist_ok=True)
            (output / 'chapter-report.json').write_text(json.dumps({'job_id': identifier, 'needs_review': review, 'pages': reports}, ensure_ascii=False, indent=2))
            source_path=root/'source.json'
            if source_path.exists() and direct_results(root,manifest,json.loads(source_path.read_text())):
                self.store.patch(identifier,state='needs_review' if review else 'completed',link=None,error=None,review=int(review),progress='النتائج جاهزة للتسليم في Discord')
                return self.store.get(identifier)
            phase = 'delivery'
            self.store.patch(identifier, state='uploading', review=int(review), progress='Uploading results to Drive')
            source=json.loads(source_path.read_text()) if source_path.exists() else {}
            import re
            folder_name=re.sub(r'[\\/\r\n\x00-\x1f]','_',source.get('name','Manhwa-'+identifier[:8]))[:150]+'-cleaned'
            folder = job['folder'] or self.drive.folder(folder_name)
            self.store.patch(identifier, folder=folder)
            delivery = bundle_results(output, root / 'delivery')
            link = self.drive.deliver(identifier, delivery, folder)
            self.store.patch(identifier, state='needs_review' if review else 'completed', link=link, progress='Delivered')
        except Exception as error:
            # Credentials and remote response bodies are never persisted.
            safe = str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__
            self.store.patch(identifier, state='delivery_failed' if phase == 'delivery' else 'failed', error=safe[:300])
        return self.store.get(identifier)
