import asyncio
from contextlib import suppress
import os
import discord
from discord import app_commands
from .sources import unpack, download_attachment
from .jobs import Store
from .drive import Drive
from .runtime import create_pipeline
from .worker import Worker

LABELS = {'queued': 'في الطابور', 'running': 'جاري التبييض', 'uploading': 'جاري الرفع',
          'completed': 'اكتملت المعالجة الآلية', 'needs_review': 'النتائج جاهزة وتحتاج مراجعة',
          'failed': 'فشلت المعالجة', 'delivery_failed': 'تمت المعالجة وفشل رفع النتائج', 'cancelled': 'ملغاة'}

def status_text(job):
    text = f"المهمة `{job['id']}`: {LABELS[job['state']]}\n{job['progress']}"
    if job['error']: text += '\n' + job['error']
    if job['link']: text += '\n' + job['link']
    if job['state'] == 'needs_review': text += '\nتجد قناع التبييض والمعاينة وتقرير النص المتبقي مع كل صفحة.'
    return text

class CleanerBot(discord.Client):
    def __init__(self, settings, store, worker):
        super().__init__(intents=discord.Intents.default(), allowed_mentions=discord.AllowedMentions.none(),
                         proxy=os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY'))
        self.tree = app_commands.CommandTree(self)
        self.settings, self.store, self.worker = settings, store, worker
        self.worker_task = None
        self.register_commands()

    async def on_ready(self):
        import logging
        print(f'Bot online: {self.user.name} ({self.user.id})', flush=True)

    async def setup_hook(self):
        guild_id = os.getenv('DISCORD_GUILD_ID')
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()
        self.worker_task = asyncio.create_task(self.work_loop())

    async def work_loop(self):
        while not self.is_closed():
            job = await asyncio.to_thread(self.worker.once)
            if job:
                # Completion uses a channel message, so long chapters do not
                # depend on an interaction token that expires in 15 minutes.
                try:
                    channel = self.get_channel(int(job['channel'])) or await self.fetch_channel(int(job['channel']))
                    await channel.send(status_text(job))
                except discord.HTTPException:
                    pass  # /status is durable even if channel permissions change.
            else:
                await asyncio.sleep(2)

    async def close(self):
        if self.worker_task:
            self.worker_task.cancel()
            with suppress(asyncio.CancelledError): await self.worker_task
        await super().close()

    def register_commands(self):
        @self.tree.command(name='clean', description='تبييض صور أو ZIP أو رابط فصل Google Drive')
        @app_commands.choices(mode=[app_commands.Choice(name='الحوار والسرد (افتراضي)', value='dialogue'),
                                   app_commands.Choice(name='كل النصوص، بما فيها المؤثرات والعلامات', value='all-text')])
        async def clean(interaction: discord.Interaction, attachment: discord.Attachment | None = None,
                        drive_url: str | None = None, mode: str = 'dialogue'):
            await interaction.response.defer(ephemeral=True, thinking=True)
            try:
                if (attachment is None) == (drive_url is None):
                    raise ValueError('اختر مرفق صورة/ZIP أو رابط درايف واحدًا.')
                if interaction.channel_id is None: raise ValueError('استخدم الأمر داخل قناة.')
                if attachment:
                    if attachment.size > self.settings.max_bytes: raise ValueError('المرفق أكبر من 50 ميغابايت.')
                    data = await asyncio.to_thread(download_attachment, attachment.url, self.settings)
                    pages = unpack(attachment.filename, data, self.settings)
                else:
                    # A separate adapter avoids racing the worker's OAuth state.
                    def read_source():
                        drive = Drive(self.settings)
                        try: return drive.read(drive_url)
                        finally: drive.close()
                    pages = await asyncio.to_thread(read_source)
                def submit():
                    from .pipeline import read_image
                    for _, data in pages: read_image(data, self.settings)
                    return self.store.enqueue(interaction.user.id, interaction.channel_id, mode, pages)
                identifier = await asyncio.to_thread(submit)
                await interaction.followup.send(f'تم تسجيل {len(pages)} صفحة. رقم المهمة: `{identifier}`.\nاستخدم `/status` لمتابعتها. سيُرسل رابط النتائج إلى هذه القناة.', ephemeral=True)
            except Exception as error:
                message = str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__
                await interaction.followup.send(message[:400], ephemeral=True)

        @self.tree.command(name='status', description='حالة مهمة التبييض ورابط نتائجها')
        async def status(interaction: discord.Interaction, job_id: str):
            try: text = status_text(self.store.get(job_id, interaction.user.id))
            except ValueError: text = 'لم تُعثر على مهمة تخصك بهذا الرقم.'
            await interaction.response.send_message(text, ephemeral=True)

        @self.tree.command(name='cancel', description='إلغاء مهمة لم تبدأ بعد')
        async def cancel(interaction: discord.Interaction, job_id: str):
            try:
                self.store.cancel(job_id, interaction.user.id); text = 'أُلغيت المهمة.'
            except ValueError as error: text = str(error)
            await interaction.response.send_message(text, ephemeral=True)

        @self.tree.command(name='retry_delivery', description='إعادة رفع نتائج محفوظة عند فشل درايف')
        async def retry(interaction: discord.Interaction, job_id: str):
            try:
                self.store.retry_delivery(job_id, interaction.user.id); text = 'أُضيف رفع النتائج إلى الطابور.'
            except ValueError as error: text = str(error)
            await interaction.response.send_message(text, ephemeral=True)


def run(settings):
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token: raise RuntimeError('DISCORD_BOT_TOKEN is missing')
    store = Store(settings)
    with store.worker_lock():
        drive = Drive(settings)
        try:
            drive.authorize()
            pipeline = create_pipeline(settings)
            bot = CleanerBot(settings, store, Worker(settings, store, pipeline, drive))
            bot.run(token)
        finally:
            drive.close()
