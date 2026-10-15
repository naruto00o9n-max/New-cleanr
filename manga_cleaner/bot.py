import asyncio
import json
import logging
import os
import time
from contextlib import suppress
from pathlib import Path
import discord
from discord import app_commands
from .sources import unpack,download_attachment,natural
from .jobs import Store,TERMINAL
from .drive import Drive
from .runtime import create_pipeline
from .worker import Worker
from .discord_ui import panel,mode_sections,source_sections,help_sections,job_sections


def status_text(job):
    return '\n'.join(job_sections(job))+ ('\n'+job['link'] if job.get('link') else '')


class ModeButtons(discord.ui.View):
    def __init__(self,bot,user_id,channel_id):
        super().__init__(timeout=900);self.bot=bot;self.user_id=user_id;self.channel_id=channel_id;self.chosen=False
    async def choose(self,interaction,label):
        if interaction.user.id!=self.user_id:
            await interaction.response.send_message('هذه اللوحة ليست لك. استخدم `/clean`.',ephemeral=True);return
        await interaction.response.defer()
        if self.chosen:return
        self.chosen=True
        await self.bot.begin_source(interaction.channel,self.user_id,label,selection_message=interaction.message)
    @discord.ui.button(label='سرعة عالية',emoji='⚡',style=discord.ButtonStyle.secondary)
    async def balanced(self,interaction,button):await self.choose(interaction,'سرعة عالية')
    @discord.ui.button(label='دقة كاملة',emoji='✅',style=discord.ButtonStyle.secondary)
    async def maximum(self,interaction,button):await self.choose(interaction,'دقة كاملة')


class CleanerBot(discord.Client):
    def __init__(self,settings,store,worker):
        intents=discord.Intents.default();intents.message_content=True
        super().__init__(intents=intents,allowed_mentions=discord.AllowedMentions.none(),proxy=os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY'))
        self.tree=app_commands.CommandTree(self);self.settings,self.store,self.worker=settings,store,worker
        self.tasks=[];self.pending={};self.last_progress={};self.register_commands()
    async def on_ready(self):
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.watching,name='Manga Bubble Cleaner | /help'))
        print(f'Bot online: {self.user.name} ({self.user.id})',flush=True)
    async def setup_hook(self):
        guild_id=os.getenv('DISCORD_GUILD_ID')
        if guild_id:
            guild=discord.Object(id=int(guild_id));self.tree.copy_global_to(guild=guild);await self.tree.sync(guild=guild)
        else:await self.tree.sync()
        self.tasks=[asyncio.create_task(self.work_loop()),asyncio.create_task(self.progress_loop())]
    async def channel_for(self,identifier):
        return self.get_channel(int(identifier)) or await self.fetch_channel(int(identifier))
    async def work_loop(self):
        while not self.is_closed():
            job=await asyncio.to_thread(self.worker.once)
            if job and not (self.settings.data/job['id']/'ui.json').exists():
                try:
                    channel=await self.channel_for(job['channel']);await self.send_final(channel,job)
                except discord.HTTPException:logging.exception('Cannot deliver job notification')
            if not job:await asyncio.sleep(2)
    async def progress_loop(self):
        while not self.is_closed():
            for path in self.settings.data.glob('*/ui.json'):
                if (path.parent/'notified.json').exists():continue
                try:
                    job=await asyncio.to_thread(self.store.get,path.parent.name)
                    channel=await self.channel_for(job['channel'])
                    if job['state'] in TERMINAL:
                        await self.send_final(channel,job)
                        (path.parent/'notified.json').write_text(json.dumps({'at':time.time()}))
                        with suppress(discord.HTTPException):await channel.get_partial_message(json.loads(path.read_text())['message_id']).delete()
                    elif time.monotonic()-self.last_progress.get(job['id'],0)>20:
                        meta=json.loads(path.read_text());message=channel.get_partial_message(meta['message_id'])
                        await message.edit(view=panel(job_sections(job)))
                        self.last_progress[job['id']]=time.monotonic()
                except (discord.HTTPException,ValueError,OSError):logging.exception('Cannot update job panel')
            await asyncio.sleep(5)
    async def send_final(self,channel,job):
        if job['state'] not in {'completed','needs_review'}:
            await channel.send(view=panel(['## ✖️ تعذرت العملية',*job_sections(job)],error=True));return
        root=self.settings.data/job['id'];report=json.loads((root/'output/chapter-report.json').read_text())
        sections=['## ✅ اكتملت معالجة الفصل',f"📷 تمت معالجة **{len(report['pages'])}** صورة."]
        if job['state']=='needs_review':sections.append('⚠️ توجد مناطق تحتاج مراجعة؛ راجع تقرير المعالجة قبل اعتماد الصفحات.')
        direct=root/'discord-delivery'
        files=sorted(direct.iterdir()) if direct.exists() else []
        if job.get('link'):
            sections.extend(['📊 وضع التسليم **Google Drive**.',f"🔗 رابط مجلد النتائج: {job['link']}"])
            await channel.send(view=panel(sections));return
        if not files:raise ValueError('لا توجد ملفات نتائج محفوظة للتسليم.')
        for start in range(0,len(files),10):
            batch=files[start:start+10];attachments=[discord.File(p,filename=p.name) for p in batch]
            try:await channel.send(view=panel(sections if start==0 else ['📷 بقية نتائج الفصل'],files=[p.name for p in batch]),files=attachments)
            finally:
                for attachment in attachments:attachment.close()
    async def close(self):
        for future in self.pending.values():future.cancel()
        for task in self.tasks:task.cancel()
        for task in self.tasks:
            with suppress(asyncio.CancelledError):await task
        await super().close()
    async def start_wizard(self,channel,user_id):
        await channel.send(view=panel(mode_sections()))
        await channel.send(content='⚙️ **اختر وضع المعالجة من الأزرار بالأسفل:**',view=ModeButtons(self,user_id,channel.id))
    async def begin_source(self,channel,user_id,label,selection_message=None):
        key=(channel.id,user_id)
        if key in self.pending:
            await channel.send(view=panel(['لديك جلسة تنتظر المصدر بالفعل. أرسل المصدر الآن.']));return
        future=asyncio.get_running_loop().create_future();self.pending[key]=future
        try:
            if selection_message is not None:await selection_message.edit(content=f'✅ تم اختيار **{label}**. أرسل المصدر في هذه القناة.',view=None)
            await channel.send(view=panel(source_sections(self.settings.max_pages)))
            message=await asyncio.wait_for(future,timeout=300)
            progress=await channel.send(view=panel(['⏳ **الحالة:** جاري قراءة المصدر','استلمت المصدر؛ يجري التحقق من الصور أو محتوى ZIP.']))
            try:
                if message.attachments:
                    attachments=message.attachments
                    if any(a.filename.lower().endswith('.zip') for a in attachments) and len(attachments)!=1:raise ValueError('أرسل ZIP واحدًا أو صورًا، دون خلطهما.')
                    pages=[];total=0
                    for a in sorted(attachments,key=lambda a:natural(a.filename)):
                        if a.size>self.settings.max_bytes:raise ValueError('حجم المرفق يتجاوز 50MB.')
                        data=await asyncio.to_thread(download_attachment,a.url,self.settings)
                        pages.extend(unpack(a.filename,data,self.settings));total+=len(data)
                        if len(pages)>self.settings.max_pages or total>self.settings.max_batch_bytes:raise ValueError('الفصل يتجاوز حدود الحجم أو عدد الصفحات.')
                    kind='zip' if attachments[0].filename.lower().endswith('.zip') else 'images'
                    name=Path(attachments[0].filename).stem if kind=='zip' else 'discord-pages'
                else:
                    def read_drive():
                        drive=Drive(self.settings)
                        try:
                            pages=drive.read(message.content.strip());return pages,drive.source_name
                        finally:drive.close()
                    pages,name=await asyncio.to_thread(read_drive);kind='drive'
                await self.submit(channel,user_id,pages,kind,name,label,progress)
            except Exception as error:
                safe=str(error) if isinstance(error,(ValueError,RuntimeError)) else type(error).__name__
                await progress.edit(view=panel(['## ✖️ تعذرت العملية',safe[:1500]],error=True))
        except asyncio.TimeoutError:await channel.send(view=panel(['انتهت مهلة انتظار المصدر. استخدم `/clean` للبدء مجددًا.'],error=True))
        finally:self.pending.pop(key,None)
    async def submit(self,channel,user_id,pages,kind,name,label,progress,mode="dialogue"):
        def enqueue():
            from .pipeline import read_image
            for _,data in pages:read_image(data,self.settings)
            return self.store.enqueue(user_id,channel.id,mode,pages,
                source={'kind':kind,'name':name,'quality_label':label,'discord_limit':min(24*1024*1024,getattr(channel.guild,'filesize_limit',10*1024*1024))},
                ui={'message_id':progress.id})
        async with self.submit_lock:
            identifier=await asyncio.to_thread(enqueue)
        await progress.edit(view=panel(['⏳ **الحالة:** في الطابور',f'📷 تم تسجيل **{len(pages)}** صورة.',f'رقم المهمة: `{identifier}`']))
        return identifier
    async def on_message(self,message):
        if message.author.bot or not message.guild:return
        future=self.pending.get((message.channel.id,message.author.id))
        if future and not future.done() and (message.attachments or message.content.strip().startswith('https://')):
            future.set_result(message);return
        command=message.content.strip().lower()
        if command in {'!clean','!تبييض'}:await self.start_wizard(message.channel,message.author.id)
        elif command in {'!help','!مساعدة','!اوامر'}:await message.channel.send(view=panel(help_sections()))
    def register_commands(self):
        self.submit_lock=asyncio.Lock()
        @self.tree.command(name='clean',description='ابدأ تبييض فصل مانهوا أو صفحة واحدة')
        @app_commands.choices(mode=[app_commands.Choice(name='الحوار والسرد',value='dialogue'),app_commands.Choice(name='كل النصوص والمؤثرات',value='all-text')])
        async def clean(interaction:discord.Interaction,attachment:discord.Attachment|None=None,drive_url:str|None=None,mode:str='dialogue'):
            if interaction.guild is None or interaction.channel is None:
                await interaction.response.send_message('استخدم `/clean` داخل قناة نصية في خادم Discord.',ephemeral=True);return
            await interaction.response.defer(ephemeral=True,thinking=True)
            if attachment is None and drive_url is None:
                await self.start_wizard(interaction.channel,interaction.user.id)
                await interaction.followup.send('أرسلت لوحة بدء التبييض في هذه القناة.',ephemeral=True);return
            try:
                if attachment and drive_url:raise ValueError('اختر مرفقًا أو رابط درايف واحدًا.')
                if attachment:
                    if attachment.size>self.settings.max_bytes:raise ValueError('المرفق أكبر من 50MB.')
                    data=await asyncio.to_thread(download_attachment,attachment.url,self.settings);pages=unpack(attachment.filename,data,self.settings)
                    kind='zip' if attachment.filename.lower().endswith('.zip') else 'images';name=Path(attachment.filename).stem
                else:
                    def read():
                        drive=Drive(self.settings)
                        try:
                            pages=drive.read(drive_url);return pages,drive.source_name
                        finally:drive.close()
                    pages,name=await asyncio.to_thread(read);kind='drive'
                progress=await interaction.channel.send(view=panel(['⏳ جاري تسجيل الفصل']))
                identifier=await self.submit(interaction.channel,interaction.user.id,pages,kind,name,'دقة كاملة',progress,mode=mode)
                await interaction.followup.send(f'تم تسجيل المهمة `{identifier}`.',ephemeral=True)
            except Exception as error:
                safe=str(error) if isinstance(error,(ValueError,RuntimeError)) else type(error).__name__
                await interaction.followup.send(safe[:1500],ephemeral=True)
        @self.tree.command(name='help',description='تعليمات استخدام Manga Bubble Cleaner')
        async def help_command(interaction:discord.Interaction):
            await interaction.response.send_message(view=panel(help_sections()),ephemeral=True)
        @self.tree.command(name='status',description='حالة مهمة التبييض ورابط نتائجها')
        async def status(interaction:discord.Interaction,job_id:str):
            try:
                job=self.store.get(job_id,interaction.user.id);await interaction.response.send_message(view=panel(job_sections(job)+([job['link']] if job.get('link') else [])),ephemeral=True)
            except ValueError:await interaction.response.send_message('لم تُعثر على مهمة تخصك بهذا الرقم.',ephemeral=True)
        @self.tree.command(name='cancel',description='إلغاء مهمة لم تبدأ بعد')
        async def cancel(interaction:discord.Interaction,job_id:str):
            try:self.store.cancel(job_id,interaction.user.id);text='أُلغيت المهمة.'
            except ValueError as error:text=str(error)
            await interaction.response.send_message(text,ephemeral=True)
        @self.tree.command(name='retry_delivery',description='إعادة رفع نتائج محفوظة عند فشل درايف')
        async def retry(interaction:discord.Interaction,job_id:str):
            try:
                self.store.retry_delivery(job_id,interaction.user.id)
                (self.settings.data/job_id/'notified.json').unlink(missing_ok=True);text='أُضيف رفع النتائج إلى الطابور.'
            except ValueError as error:text=str(error)
            await interaction.response.send_message(text,ephemeral=True)


def run(settings):
    token=os.getenv('DISCORD_BOT_TOKEN')
    if not token:raise RuntimeError('DISCORD_BOT_TOKEN is missing')
    store=Store(settings)
    with store.worker_lock():
        drive=Drive(settings)
        try:
            drive.authorize();pipeline=create_pipeline(settings)
            CleanerBot(settings,store,Worker(settings,store,pipeline,drive)).run(token.strip())
        finally:drive.close()
