import io
import zipfile
import pytest
from manga_cleaner.config import Settings
from manga_cleaner.sources import unpack, drive_id, download_attachment

def archive(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in entries: z.writestr(name, data)
    return stream.getvalue()

def test_zip_chapter_natural_order():
    pages = unpack('chapter.zip', archive([('10.jpg', b'a'), ('2.jpg', b'b'), ('1.jpg', b'c')]), Settings())
    assert [name for name, _ in pages] == ['1.jpg', '2.jpg', '10.jpg']

@pytest.mark.parametrize('name', ['../page.jpg', '/page.jpg', '..\\page.jpg'])
def test_zip_traversal_rejected(name):
    with pytest.raises(ValueError): unpack('chapter.zip', archive([(name, b'a')]), Settings())

def test_zip_bomb_rejected():
    with pytest.raises(ValueError): unpack('chapter.zip', archive([('1.jpg', b'a'*100000)]), Settings())

def test_drive_links_and_untrusted_sources():
    assert drive_id('https://drive.google.com/drive/folders/abcdefghijk') == 'abcdefghijk'
    with pytest.raises(ValueError): drive_id('https://evil.example/drive/folders/abcdefghijk')
    with pytest.raises(ValueError): download_attachment('http://127.0.0.1/internal', Settings())
