import json
import time
import httpx
import pytest
from manga_cleaner.drive import Drive
from manga_cleaner.config import Settings


def adapter(handler):
    d = Drive(Settings(), httpx.Client(transport=httpx.MockTransport(handler)))
    d.token, d.expires = 'test-token', time.monotonic()+3600
    return d


def test_resumable_upload_chunks_and_acknowledgments(tmp_path):
    path = tmp_path/'cleaned.png'; path.write_bytes(b'x'*(8*1024*1024+10))
    ranges = []
    def handle(request):
        if request.method == 'POST':
            assert request.url.params['uploadType'] == 'resumable'
            return httpx.Response(200, headers={'Location':'https://www.googleapis.com/upload/session'})
        ranges.append(request.headers['Content-Range'])
        if len(ranges) == 1:
            return httpx.Response(308, headers={'Range':f'bytes=0-{8*1024*1024-1}'})
        return httpx.Response(200, json={'id':'uploaded'})
    d=adapter(handle)
    assert d.upload(path,'folder') == 'uploaded'
    assert len(ranges) == 2
    assert ranges[1] == f'bytes {8*1024*1024}-{path.stat().st_size-1}/{path.stat().st_size}'


def test_delivery_skips_existing_files_and_sets_link_permission(tmp_path):
    (tmp_path/'cleaned.png').write_bytes(b'data')
    calls=[]
    def handle(request):
        calls.append((request.method,str(request.url)))
        if request.method == 'GET': return httpx.Response(200,json={'files':[{'name':'cleaned.png'}]})
        assert request.url.path.endswith('/permissions')
        assert json.loads(request.content) == {'type':'anyone','role':'reader'}
        return httpx.Response(200,json={'id':'permission'})
    d=adapter(handle)
    assert d.deliver('job',tmp_path,'folder') == 'https://drive.google.com/drive/folders/folder'
    assert len(calls) == 2


def test_drive_rejects_oversized_source_without_download():
    def handle(request):
        assert request.url.params.get('alt') is None
        return httpx.Response(200,json={'id':'abcdefghijk','name':'p.jpg','mimeType':'image/jpeg','size':str(100*1024*1024)})
    with pytest.raises(ValueError): adapter(handle).read('https://drive.google.com/file/d/abcdefghijk/view')
