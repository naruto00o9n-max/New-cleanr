from dataclasses import replace
import json
import pytest
from manga_cleaner.config import Settings
from manga_cleaner.jobs import Store
from manga_cleaner.worker import Worker

@pytest.fixture
def setup(tmp_path):
    settings = replace(Settings(), data=tmp_path)
    return settings, Store(settings)

def test_persistence_ownership_queue_and_recovery(setup):
    settings, store = setup
    identifier = store.enqueue('alice', 'channel', 'dialogue', [('page.jpg', b'input')])
    with pytest.raises(ValueError): store.enqueue('alice', 'channel', 'dialogue', [('page.jpg', b'input')])
    with pytest.raises(ValueError): store.get(identifier, 'bob')
    assert Store(settings).get(identifier)['state'] == 'queued'
    with store.worker_lock():
        assert store.claim()['id'] == identifier
        assert store.claim() is None
        with pytest.raises(RuntimeError):
            with Store(settings).worker_lock(): pass
    with Store(settings).worker_lock(): assert store.get(identifier)['state'] == 'queued'


def test_cancellation_before_claim(setup):
    settings, store = setup
    identifier = store.enqueue('alice', 'channel', 'dialogue', [('page.jpg', b'input')])
    store.cancel(identifier, 'alice')
    assert store.claim() is None


def test_upload_retry_reuses_processed_page_and_folder(setup):
    settings, store = setup
    class Pipeline:
        calls = 0
        def clean(self, data, target, mode, progress):
            self.calls += 1; target.mkdir(parents=True)
            report = {'status':'completed'}
            (target/'cleaned.png').write_bytes(b'result')
            (target/'report.json').write_text(json.dumps(report))
            return report
    class Drive:
        fails = True
        folders = 0
        def folder(self, name): self.folders += 1; return 'folder'
        def deliver(self, identifier, output, folder):
            assert folder == 'folder'
            if self.fails: raise RuntimeError('upload unavailable')
            return 'https://drive.google.com/drive/folders/folder'
    pipeline, drive = Pipeline(), Drive()
    identifier = store.enqueue('alice', 'channel', 'all-text', [('page.jpg', b'input')])
    worker = Worker(settings, store, pipeline, drive)
    with store.worker_lock():
        assert worker.once()['state'] == 'delivery_failed'
        drive.fails = False
        store.retry_delivery(identifier, 'alice')
        assert worker.once()['state'] == 'completed'
    assert pipeline.calls == drive.folders == 1


def test_processing_failure_releases_next_job(setup):
    settings, store = setup
    class Broken:
        def clean(self, *args): raise RuntimeError('model failed')
    a = store.enqueue('alice', 'channel', 'dialogue', [('p.jpg', b'input')])
    b = store.enqueue('bob', 'channel', 'dialogue', [('p.jpg', b'input')])
    worker = Worker(settings, store, Broken(), None)
    with store.worker_lock():
        assert worker.once()['id'] == a
        assert worker.once()['id'] == b
    assert store.get(a)['state'] == 'failed'


def test_delivery_bundle_separates_clean_pages_from_masks(tmp_path):
    import zipfile
    from manga_cleaner.worker import bundle_results
    output=tmp_path/'output';page=output/'0001';page.mkdir(parents=True)
    (page/'cleaned.png').write_bytes(b'clean')
    (page/'mask.png').write_bytes(b'mask')
    (page/'overlay.png').write_bytes(b'overlay')
    (page/'report.json').write_text('{}')
    (output/'chapter-report.json').write_text('{}')
    result=bundle_results(output,tmp_path/'delivery')
    assert (result/'0001.png').read_bytes() == b'clean'
    assert not (result/'mask.png').exists()
    with zipfile.ZipFile(result/'review-artifacts.zip') as archive:
        assert '0001/mask.png' in archive.namelist()
        assert '0001/cleaned.png' not in archive.namelist()
