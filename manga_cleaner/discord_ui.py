"""Original gold/red Components V2 panels and input wizard copy."""
import discord

GOLD=0xD5AA55
RED=0xD94A4A


def panel(sections,error=False,files=()):
    view=discord.ui.LayoutView(timeout=None)
    children=[]
    for section in filter(None,sections):
        if children:children.append(discord.ui.Separator())
        children.append(discord.ui.TextDisplay(section))
    for name in files:
        children.extend([discord.ui.Separator(),discord.ui.File('attachment://'+name)])
    view.add_item(discord.ui.Container(*children,accent_colour=RED if error else GOLD))
    return view


def mode_sections():
    return ['## ⚙️ اختر وضع التبييض',
            '### ⚡ سرعة عالية\nتبييض تلقائي لصور الفصل مع الحفاظ على الرسم.',
            '### ✅ دقة كاملة\nمعالجة الفقاعات البيضاء والداكنة والملونة بمحرك التبييض المحسّن.']


def source_sections(max_pages=100):
    return ['## ⏳ في انتظار المصدر — لم تبدأ المعالجة بعد',
            'أرسل الصورة أو ZIP الآن في القناة؛ ستظهر لوحة التقدم فقط عند وصول مصدر صالح.',
            '## 📂 خطوة إرفاق صور الفصل\nتوجد ثلاث طرق لإرسال صور فصلك.',
            f'📷 **الطريقة الأولى:** أرفق حتى **{max_pages}** صورة مباشرة، مرتبة بأسمائها.',
            '🗜️ **الطريقة الثانية:** أرسل ZIP واحدًا حتى **50MB**؛ سيعود ZIP بالنتائج، أو رابط Drive إن تجاوز حد المرفقات.',
            '🗂️ **الطريقة الثالثة:** أرسل رابط Google Drive لصورة أو ZIP أو مجلد؛ سيعود رابط مجلد نتائج جديد. الروابط العامة مدعومة، والمصادر الخاصة تحتاج صلاحية API لقراءتها.']


def help_sections():
    return ['# 📷 Manga Bubble Cleaner',
            'بوت لتبييض نصوص فقاعات المانهوا مع الحفاظ على الرسم. استخدم `/clean` أو `!clean` للبدء.',
            '🎮 **طريقة العمل**\nاختر سرعة عالية أو دقة كاملة، ثم أرسل صورًا مباشرة أو ZIP أو رابط Google Drive داخل القناة نفسها.',
            '📦 **تسليم النتيجة**\nالمرفقات تعود صورًا، ZIP يعود ZIP، وGoogle Drive يعود برابط مجلد نتائج جديد. المهام محفوظة لاستكمالها بعد انقطاع التشغيل.']


def job_sections(job):
    states={'queued':'في الطابور','running':'التبييض مستمر','uploading':'تجهيز النتيجة','failed':'✖️ تعذرت العملية','delivery_failed':'✖️ تعذر رفع النتيجة','cancelled':'ألغيت المهمة'}
    return [f"⏳ **الحالة:** {states.get(job['state'],job['state'])}",
            f"📊 **التقدم:** {job['progress'] or 'انتظار بدء المعالجة'}",
            f"ℹ️ {job['error']}" if job.get('error') else f"رقم المهمة: `{job['id']}`"]
