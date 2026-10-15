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
        assert json.loads(request.content) == {'type':'anyone','role':'reader','allowFileDiscovery':False}
        return httpx.Response(200,json={'id':'permission'})
    d=adapter(handle)
    assert d.deliver('job',tmp_path,'folder') == 'https://drive.google.com/drive/folders/folder'
    assert len(calls) == 2


def test_drive_rejects_oversized_source_without_download():
    def handle(request):
        assert request.url.params.get('alt') is None
        return httpx.Response(200,json={'id':'abcdefghijk','name':'p.jpg','mimeType':'image/jpeg','size':str(100*1024*1024)})
    with pytest.raises(ValueError): adapter(handle).read('https://drive.google.com/file/d/abcdefghijk/view')


def public_html(rows):
    # Drive wraps JSON in a JS single-quoted string with hexadecimal escapes.
    data=json.dumps([rows]).replace('"','\\x22').replace('[','\\x5b').replace(']','\\x5d')
    return "<script>window['_DRIVE_ivd'] = '"+data+"';</script>"


def test_api_404_reads_public_folder_without_forwarding_oauth():
    def handle(request):
        if request.url.host=='www.googleapis.com':
            assert request.url.params['supportsAllDrives']=='true'
            return httpx.Response(404,json={'error':{'code':404}})
        assert 'Authorization' not in request.headers
        if '/folders/' in request.url.path:
            return httpx.Response(200,text=public_html([['page2id',[],'2.jpg','image/jpeg'],['page1id',[],'1.jpg','image/jpeg']]))
        return httpx.Response(200,content=request.url.params['id'].encode(),headers={'content-type':'image/jpeg'})
    pages=adapter(handle).read('https://drive.google.com/drive/folders/abcdefghijk')
    assert pages==[('1.jpg',b'page1id'),('2.jpg',b'page2id')]


def test_public_reader_rejects_private_folder_and_partial_large_listing():
    from manga_cleaner.public_drive import folder_entries
    with pytest.raises(ValueError,match='المجلد العام'):folder_entries('<title>Sign in</title>',100)
    with pytest.raises(ValueError,match='كبير'):folder_entries(public_html([[str(i),[],'1.jpg','image/jpeg'] for i in range(50)]),100)


def test_public_download_confirmation_and_redirect_restrictions():
    from manga_cleaner.public_drive import PublicDrive
    def handle(request):
        if request.url.host=='drive.google.com':
            return httpx.Response(200,text='<form action="https://drive.usercontent.google.com/download"><input name="id" value="page"><input name="confirm" value="yes"></form>',headers={'content-type':'text/html'})
        assert request.url.params['confirm']=='yes'
        return httpx.Response(200,content=b'image',headers={'content-type':'image/jpeg'})
    assert PublicDrive(httpx.Client(transport=httpx.MockTransport(handle)),Settings()).file('page')==b'image'
    def malicious(request):return httpx.Response(302,headers={'Location':'https://example.com/file'})
    with pytest.raises(ValueError,match='redirect'):
        PublicDrive(httpx.Client(transport=httpx.MockTransport(malicious)),Settings()).file('page')
