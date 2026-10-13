import json
from pathlib import Path
from .pipeline import read_image

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
            phase = 'delivery'
            self.store.patch(identifier, state='uploading', review=int(review), progress='Uploading results to Drive')
            folder = job['folder'] or self.drive.folder('Manhwa-' + identifier[:8])
            self.store.patch(identifier, folder=folder)
            link = self.drive.deliver(identifier, output, folder)
            self.store.patch(identifier, state='needs_review' if review else 'completed', link=link, progress='Delivered')
        except Exception as error:
            # Credentials and remote response bodies are never persisted.
            safe = str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__
            self.store.patch(identifier, state='delivery_failed' if phase == 'delivery' else 'failed', error=safe[:300])
        return self.store.get(identifier)
