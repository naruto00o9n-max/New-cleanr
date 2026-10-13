"""Single process lease and transactional SQLite queue, independent of MongoDB."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sqlite3
import time
import uuid

TERMINAL = ('completed', 'needs_review', 'failed', 'delivery_failed', 'cancelled')

class Store:
    def __init__(self, settings):
        self.settings = settings
        settings.data.mkdir(parents=True, exist_ok=True)
        self.path = settings.data / 'jobs.sqlite3'
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, channel TEXT NOT NULL,
                mode TEXT NOT NULL, state TEXT NOT NULL, progress TEXT NOT NULL DEFAULT '',
                created REAL NOT NULL, updated REAL NOT NULL, error TEXT,
                folder TEXT, link TEXT, review INTEGER NOT NULL DEFAULT 0)''')
            db.execute('CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state, created)')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def get(self, identifier, owner=None):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone()
        if row is None or (owner is not None and row['owner'] != str(owner)):
            raise ValueError('Job not found')
        return dict(row)

    def patch(self, identifier, **values):
        if not set(values) <= {'state', 'progress', 'error', 'folder', 'link', 'review'}:
            raise ValueError('Unknown job fields')
        values['updated'] = time.time()
        with self.connect() as db:
            db.execute('UPDATE jobs SET '+','.join(f'{key}=?' for key in values)+' WHERE id=?', (*values.values(), identifier))

    def enqueue(self, owner, channel, mode, pages):
        if mode not in {'dialogue', 'all-text'}: raise ValueError('Invalid cleaning mode')
        if not pages or len(pages) > self.settings.max_pages or sum(len(data) for _, data in pages) > self.settings.max_batch_bytes:
            raise ValueError('Invalid chapter size')
        if any(len(data) > self.settings.max_bytes for _, data in pages):
            raise ValueError('Page exceeds byte limit')
        identifier = uuid.uuid4().hex
        directory = self.settings.data / identifier / 'input'
        try:
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                count = db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running','uploading')").fetchone()[0]
                if count >= self.settings.queue_limit: raise ValueError('Queue is full')
                count = db.execute("SELECT COUNT(*) FROM jobs WHERE owner=? AND state IN ('queued','running','uploading')", (str(owner),)).fetchone()[0]
                if count: raise ValueError('You already have an active chapter')
                directory.mkdir(parents=True)
                metadata = []
                for index, (name, data) in enumerate(pages, 1):
                    path = directory / f'{index:04d}.image'
                    path.write_bytes(data)
                    metadata.append({'path': path.name, 'source_name': name})
                (directory / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False))
                now = time.time()
                db.execute('INSERT INTO jobs(id,owner,channel,mode,state,created,updated) VALUES(?,?,?,?,?,?,?)',
                           (identifier, str(owner), str(channel), mode, 'queued', now, now))
        except Exception:
            import shutil
            shutil.rmtree(directory.parent, ignore_errors=True)
            raise
        return identifier

    @contextmanager
    def worker_lock(self):
        with (self.settings.data / 'worker.lock').open('a') as handle:
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('A cleaning worker is already running')
            try:
                # Only the lease owner may recover interrupted work.
                with self.connect() as db:
                    db.execute("UPDATE jobs SET state='queued',updated=? WHERE state IN ('running','uploading')", (time.time(),))
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def claim(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET state='running',updated=? WHERE id=?", (time.time(), row['id']))
                return dict(row)

    def cancel(self, identifier, owner):
        self.get(identifier, owner)
        with self.connect() as db:
            result = db.execute("UPDATE jobs SET state='cancelled',updated=? WHERE id=? AND state='queued'", (time.time(), identifier))
        if not result.rowcount: raise ValueError('Only queued jobs can be cancelled')

    def retry_delivery(self, identifier, owner):
        self.get(identifier, owner)
        with self.connect() as db:
            result = db.execute("UPDATE jobs SET state='queued',error=NULL,updated=? WHERE id=? AND state='delivery_failed'", (time.time(), identifier))
        if not result.rowcount: raise ValueError('This job has no failed delivery')

    def expire(self, days=7):
        import shutil
        threshold = time.time()-days*86400
        with self.connect() as db:
            rows = db.execute("SELECT id FROM jobs WHERE updated<? AND state IN ('completed','needs_review','failed','delivery_failed','cancelled')", (threshold,)).fetchall()
            for row in rows:
                shutil.rmtree(self.settings.data / row['id'], ignore_errors=True)
                db.execute('DELETE FROM jobs WHERE id=?', (row['id'],))
        return len(rows)
