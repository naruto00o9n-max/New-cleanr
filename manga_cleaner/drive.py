"""Drive REST adapter: bounded imports, resumable uploads, explicit delivery."""
import os
import time
from pathlib import Path
import httpx
from .sources import drive_id, unpack, EXTENSIONS, natural

class DriveError(RuntimeError):
    def __init__(self,status):
        self.status=status
        super().__init__(f"Drive API failed (HTTP {status})")


class Drive:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client or httpx.Client(timeout=120, follow_redirects=False)
        self.token, self.expires = '', 0
        self.source_name='drive-folder'

    def close(self):
        self.client.close()

    def authorize(self):
        if time.monotonic() < self.expires:
            return
        names = ('GDRIVE_CLIENT_ID', 'GDRIVE_CLIENT_SECRET', 'GDRIVE_REFRESH_TOKEN')
        if not all(os.getenv(name) for name in names):
            raise RuntimeError('Google Drive credentials are incomplete')
        response = self.client.post('https://oauth2.googleapis.com/token', data={
            'client_id': os.environ[names[0]], 'client_secret': os.environ[names[1]],
            'refresh_token': os.environ[names[2]], 'grant_type': 'refresh_token'})
        if response.status_code != 200:
            raise RuntimeError(f'Drive OAuth failed (HTTP {response.status_code})')
        body = response.json()
        self.token = body['access_token']
        self.expires = time.monotonic() + max(0, int(body['expires_in'])-60)

    def request(self, method, path, **kwargs):
        self.authorize()
        if path.startswith('files'):
            kwargs['params']={**kwargs.get('params',{}),'supportsAllDrives':'true'}
        response = self.client.request(method, 'https://www.googleapis.com/drive/v3/' + path,
                                       headers={'Authorization': 'Bearer ' + self.token}, **kwargs)
        if response.status_code == 401:
            self.expires = 0; self.authorize()
            response = self.client.request(method, 'https://www.googleapis.com/drive/v3/' + path,
                                           headers={'Authorization': 'Bearer ' + self.token}, **kwargs)
        if not response.is_success:
            raise DriveError(response.status_code)
        return response

    def read(self, url):
        identifier=drive_id(url)
        try:return self._read_api(url)
        except DriveError as error:
            if error.status not in {403,404}:raise
            from .public_drive import PublicDrive
            try:
                public=PublicDrive(self.client,self.settings)
                pages=public.read(url,identifier);self.source_name=public.source_name
                return pages
            except ValueError as public_error:
                raise ValueError(str(public_error)+' المصادر الخاصة تحتاج ربط Drive بصلاحية قراءة المصدر؛ يمكنك إرسال ZIP أو رابط عام قابل للتنزيل.') from None

    def _read_api(self, url):
        identifier = drive_id(url)
        info = self.request('GET', f'files/{identifier}', params={'fields': 'id,name,mimeType,size','supportsAllDrives':'true'}).json()
        self.source_name=info.get('name','drive-folder')
        if info['mimeType'] == 'application/vnd.google-apps.folder':
            entries, cursor = [], None
            while True:
                params = {'q': f"'{identifier}' in parents and trashed = false", 'fields': 'nextPageToken,files(id,name,mimeType,size)', 'pageSize': 100,'supportsAllDrives':'true','includeItemsFromAllDrives':'true'}
                if cursor: params['pageToken'] = cursor
                body = self.request('GET', 'files', params=params).json()
                entries.extend(f for f in body['files'] if Path(f['name']).suffix.lower() in EXTENSIONS | {'.zip'})
                if len(entries) > self.settings.max_pages:
                    raise ValueError('Drive folder exceeds page limit')
                cursor = body.get('nextPageToken')
                if not cursor: break
        else:
            entries = [info]
        pages, total = [], 0
        for entry in sorted(entries, key=lambda f: natural(f['name'])):
            if int(entry.get('size', self.settings.max_bytes+1)) > self.settings.max_bytes:
                raise ValueError('Drive file exceeds limit')
            self.authorize()
            with self.client.stream('GET', f"https://www.googleapis.com/drive/v3/files/{entry['id']}",
                                    params={'alt': 'media','supportsAllDrives':'true'}, headers={'Authorization': 'Bearer '+self.token}) as response:
                if not response.is_success: raise DriveError(response.status_code)
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > self.settings.max_bytes: raise ValueError('Drive download exceeds limit')
                    chunks.append(chunk)
            new = unpack(entry['name'], b''.join(chunks), self.settings)
            total += sum(len(data) for _, data in new)
            pages.extend(new)
            if len(pages) > self.settings.max_pages or total > self.settings.max_batch_bytes:
                raise ValueError('Drive chapter exceeds limits')
        if not pages: raise ValueError('Drive source contains no supported images')
        return pages

    def folder(self, name):
        body = {'name': name, 'mimeType': 'application/vnd.google-apps.folder'}
        if os.getenv('GDRIVE_PARENT_FOLDER_ID'): body['parents'] = [os.environ['GDRIVE_PARENT_FOLDER_ID']]
        return self.request('POST', 'files', json=body, params={'fields': 'id'}).json()['id']

    def upload(self, path, folder, name=None):
        self.authorize()
        metadata = {'name': name or path.name, 'parents': [folder]}
        size = path.stat().st_size
        response = self.client.post('https://www.googleapis.com/upload/drive/v3/files',
                                    params={'uploadType': 'resumable','supportsAllDrives':'true'}, json=metadata,
                                    headers={'Authorization': 'Bearer '+self.token, 'X-Upload-Content-Length': str(size)})
        if not response.is_success: raise RuntimeError(f'Drive upload initialization failed (HTTP {response.status_code})')
        url = response.headers['Location']
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname != 'www.googleapis.com':
            raise RuntimeError('Invalid Drive upload session URL')
        chunk_size = 8 * 1024 * 1024
        with path.open('rb') as stream:
            offset = 0
            while offset < size:
                chunk = stream.read(chunk_size)
                for attempt in range(4):
                    try:
                        uploaded = self.client.put(url, content=chunk, headers={
                            'Content-Length': str(len(chunk)), 'Content-Range': f'bytes {offset}-{offset+len(chunk)-1}/{size}'})
                        if uploaded.status_code in {200, 201, 308}: break
                        if uploaded.status_code < 500 and uploaded.status_code != 429:
                            raise RuntimeError(f'Drive upload failed (HTTP {uploaded.status_code})')
                    except httpx.TransportError:
                        if attempt == 3: raise
                    if attempt == 3: raise RuntimeError('Drive upload retry limit exceeded')
                    time.sleep(2 ** attempt)
                if uploaded.status_code == 308:
                    acknowledged = uploaded.headers.get('Range', '')
                    if acknowledged != f'bytes=0-{offset+len(chunk)-1}':
                        raise RuntimeError('Drive upload acknowledgment mismatch')
                offset += len(chunk)
        if uploaded.status_code not in {200, 201}:
            raise RuntimeError('Drive upload did not complete')
        return uploaded.json()['id']

    def deliver(self, job_id, output, existing_folder=None):
        folder = existing_folder or self.folder('Manhwa-' + job_id[:8])
        # Stable filenames plus listing make upload retries idempotent.
        existing, cursor = set(), None
        while True:
            params = {'q': f"'{folder}' in parents and trashed = false", 'fields': 'files(name),nextPageToken', 'pageSize': 1000}
            if cursor: params['pageToken'] = cursor
            body = self.request('GET', 'files', params=params).json()
            existing.update(f['name'] for f in body['files'])
            cursor = body.get('nextPageToken')
            if not cursor: break
        for path in sorted(output.rglob('*')):
            if path.is_file():
                name = str(path.relative_to(output)).replace('/', '__')
                if name not in existing: self.upload(path, folder, name)
        self.request('POST', f'files/{folder}/permissions', json={'type': 'anyone', 'role': 'reader','allowFileDiscovery':False},params={'sendNotificationEmail':'false'})
        return f'https://drive.google.com/drive/folders/{folder}'
