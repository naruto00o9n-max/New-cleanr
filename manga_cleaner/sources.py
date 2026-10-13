import io
from pathlib import PurePosixPath
import re
import zipfile
from urllib.parse import urlparse
import httpx

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

def natural(value):
    return [int(v) if v.isdigit() else v.casefold() for v in re.split(r'(\d+)', value)]

def unpack(name, data, settings):
    if len(data) > settings.max_bytes:
        raise ValueError('Input exceeds 50 MB')
    if not name.lower().endswith('.zip'):
        if PurePosixPath(name).suffix.lower() not in EXTENSIONS:
            raise ValueError('Unsupported file type')
        return [(PurePosixPath(name).name, data)]
    pages, total = [], 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = [e for e in archive.infolist() if not e.is_dir() and PurePosixPath(e.filename).suffix.lower() in EXTENSIONS and '__MACOSX' not in e.filename]
        if not entries or len(entries) > settings.max_pages:
            raise ValueError('ZIP must contain 1–100 images')
        for entry in sorted(entries, key=lambda e: natural(e.filename)):
            path = PurePosixPath(entry.filename.replace('\\', '/'))
            if path.is_absolute() or '..' in path.parts or entry.flag_bits & 1:
                raise ValueError('Unsafe or encrypted ZIP entry')
            total += entry.file_size
            if entry.file_size > settings.max_bytes or total > settings.max_batch_bytes or entry.file_size > max(1, entry.compress_size) * 200:
                raise ValueError('ZIP exceeds decompression limits')
            pages.append((path.name, archive.read(entry)))
    return pages


def download_attachment(url, settings):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in {'cdn.discordapp.com', 'media.discordapp.net'}:
        raise ValueError('Use a Discord attachment')
    with httpx.Client(timeout=120, follow_redirects=False) as client:
        with client.stream('GET', url) as response:
            response.raise_for_status()
            chunks, total = [], 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > settings.max_bytes:
                    raise ValueError('Attachment exceeds byte limit')
                chunks.append(chunk)
    return b''.join(chunks)


def drive_id(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'drive.google.com':
        raise ValueError('Use a drive.google.com file or folder link')
    match = re.search(r'/(?:folders|d)/([A-Za-z0-9_-]+)', parsed.path)
    if not match:
        from urllib.parse import parse_qs
        value = parse_qs(parsed.query).get('id', [''])[0]
    else:
        value = match[1]
    if not re.fullmatch(r'[A-Za-z0-9_-]{10,}', value):
        raise ValueError('Invalid Drive link')
    return value
