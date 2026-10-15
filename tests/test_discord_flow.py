import asyncio
import io
import json
import zipfile
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from PIL import Image
from manga_cleaner.config import Settings
from manga_cleaner.jobs import Store
from manga_cleaner.worker import Worker,direct_results
from manga_cleaner.bot import CleanerBot,ModeButtons
from manga_cleaner.discord_ui import panel,mode_sections,source_sections


def test_original_gold_v2_panel_and_mode_buttons():
    async def run():
        view=panel(mode_sections());payload=view.to_components()
        assert payload[0]['type']==17 and payload[0]['accent_color']==0xD5AA55
        assert panel(['failed'],error=True).to_components()[0]['accent_color']==0xD94A4A
        bot=SimpleNamespace(begin_source=AsyncMock())
        buttons=ModeButtons(bot,1,9)
        assert [b.label for b in buttons.children]==['سرعة عالية','دقة كاملة']
        wrong=SimpleNamespace(user=SimpleNamespace(id=2),response=SimpleNamespace(send_message=AsyncMock()))
        await buttons.choose(wrong,'دقة كاملة');assert not buttons.chosen;bot.begin_source.assert_not_called()
        right=SimpleNamespace(user=SimpleNamespace(id=1),response=SimpleNamespace(defer=AsyncMock()),message=SimpleNamespace(edit=AsyncMock()),channel='channel')
        await buttons.choose(right,'دقة كاملة')
        right.response.defer.assert_awaited_once();bot.begin_source.assert_awaited_once_with('channel',1,'دقة كاملة',selection_message=right.message)
    asyncio.run(run())


def test_waiting_source_is_scoped_to_user_and_channel(tmp_path):
    async def run():
        settings=replace(Settings(),data=tmp_path);bot=CleanerBot(settings,Store(settings),None)
        future=asyncio.get_running_loop().create_future();bot.pending[(10,1)]=future
        def message(user,channel):return SimpleNamespace(author=SimpleNamespace(bot=False,id=user),guild=True,channel=SimpleNamespace(id=channel),attachments=[],content='https://drive.google.com/drive/folders/abcdefghijk')
        await bot.on_message(message(2,10));assert not future.done()
        await bot.on_message(message(1,11));assert not future.done()
        good=message(1,10);await bot.on_message(good);assert future.result() is good
        assert {'clean','help','status','cancel','retry_delivery'}=={c.name for c in bot.tree.get_commands()}
        await bot.close()
    asyncio.run(run())


def test_enqueue_commits_source_and_ui_before_job_is_claimable(tmp_path):
    store=Store(replace(Settings(),data=tmp_path))
    identifier=store.enqueue(1,2,'dialogue',[('1.jpg',b'data')],source={'kind':'images'},ui={'message_id':3})
    assert store.claim()['id']==identifier
    assert json.loads((tmp_path/identifier/'source.json').read_text())['kind']=='images'
    assert json.loads((tmp_path/identifier/'ui.json').read_text())['message_id']==3


def test_original_image_and_zip_delivery_and_large_file_fallback(tmp_path):
    root=tmp_path/'job';out=root/'output'/'0001';out.mkdir(parents=True);(out/'cleaned.png').write_bytes(b'cleaned')
    manifest=[{'source_name':'1.jpg'}]
    assert direct_results(root,manifest,{'kind':'images','discord_limit':1000})
    assert (root/'discord-delivery/1-clean.png').read_bytes()==b'cleaned'
    assert direct_results(root,manifest,{'kind':'zip','name':'chapter','discord_limit':1000})
    with zipfile.ZipFile(root/'discord-delivery/chapter-cleaned.zip') as z:assert z.read('1-clean.png')==b'cleaned'
    assert not direct_results(root,manifest,{'kind':'images','discord_limit':1})
    assert not (root/'discord-delivery').exists()
    assert not direct_results(root,manifest,{'kind':'drive'})


def test_worker_direct_images_do_not_require_drive_upload(tmp_path):
    settings=replace(Settings(),data=tmp_path);store=Store(settings)
    identifier=store.enqueue(1,2,'dialogue',[('1.jpg',b'input')],source={'kind':'images','discord_limit':1000})
    class Pipeline:
        def clean(self,data,target,*args):
            target.mkdir(parents=True);(target/'cleaned.png').write_bytes(b'output');return {'status':'completed'}
    job=Worker(settings,store,Pipeline(),None).once()
    assert job['state']=='completed' and not job['link']
    assert (tmp_path/identifier/'discord-delivery/1-clean.png').read_bytes()==b'output'


def test_duplicate_source_names_never_overwrite_cleaned_pages(tmp_path):
    manifest=[{'source_name':name} for name in ['x-3.jpg','x.jpg','x.jpg']]
    for index in range(1,4):
        target=tmp_path/'output'/f'{index:04d}';target.mkdir(parents=True);(target/'cleaned.png').write_bytes(bytes([index]))
    assert direct_results(tmp_path,manifest,{'kind':'images','discord_limit':1000})
    files=list((tmp_path/'discord-delivery').iterdir())
    assert len(files)==3 and {p.read_bytes() for p in files}=={b'\x01',b'\x02',b'\x03'}
