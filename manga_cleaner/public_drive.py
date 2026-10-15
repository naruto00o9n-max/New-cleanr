"""Read only publicly shared Drive resources when drive.file cannot see them."""
import json
from html import unescape
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urljoin
from .sources import EXTENSIONS, natural, unpack


def google_download_host(url):
    p=urlparse(url)
    return p.scheme=='https' and (p.hostname in {'drive.google.com','drive.usercontent.google.com'} or (p.hostname or '').endswith('.googleusercontent.com'))


def folder_entries(html, limit):
    found=re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'((?:\\.|[^'\\])*)'",html)
    if not found:raise ValueError('تعذر قراءة المجلد العام. تأكد من أن المشاركة تسمح لأي شخص لديه الرابط بالعرض.')
    def escape(m):
        v=m.group(1)
        if v.startswith(('x','u')) and len(v)>1:return chr(int(v[1:],16))
        return {'n':'\n','r':'\r','t':'\t','b':'\b','f':'\f'}.get(v,v)
    decoded=re.sub(r'\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)',escape,found.group(1))
    try:raw=json.loads(decoded)[0] or []
    except (ValueError,IndexError,TypeError):raise ValueError('تغيرت صيغة قائمة Google Drive العامة؛ أرسل ZIP أو شارك المجلد مع حساب البوت.') from None
    # The public reader may paginate at 50 files. Never silently clean a subset.
    if len(raw)>=50:raise ValueError('المجلد العام كبير؛ شاركه مع حساب البوت للوصول إلى جميع الصفحات، أو أرسل ZIP.')
    entries=[]
    for row in raw:
        if not isinstance(row,list) or len(row)<4:continue
        identifier,name,mime=row[0],row[2],row[3]
        if not isinstance(identifier,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',identifier):continue
        if not isinstance(name,str) or Path(name).suffix.lower() not in EXTENSIONS|{'.zip'}:continue
        entries.append({'id':identifier,'name':name,'mimeType':mime})
    if len(entries)>limit:raise ValueError('Drive folder exceeds page limit')
    return sorted(entries,key=lambda f:natural(f['name']))


class DownloadForm(HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.fields={};self.active=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and self.action is None:
            self.action=a.get('action');self.active=True
        if tag=='input' and self.active and a.get('name'):self.fields[a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':self.active=False


class PublicDrive:
    def __init__(self,client,settings):self.client,self.settings=client,settings;self.source_name="drive-folder"
    def fetch(self,url,params=None,limit=None):
        limit=self.settings.max_bytes if limit is None else limit
        for _ in range(6):
            if not google_download_host(url):raise ValueError('Invalid public Drive download redirect')
            with self.client.stream('GET',url,params=params,follow_redirects=False) as r:
                if r.status_code in {301,302,303,307,308}:
                    url=urljoin(str(r.url),r.headers.get('Location',''));params=None;continue
                if not r.is_success:raise ValueError('تعذر تنزيل المصدر العام من Google Drive. تحقق من المشاركة وصلاحية التنزيل.')
                chunks=[];size=0
                for chunk in r.iter_bytes():
                    size+=len(chunk)
                    if size>limit:raise ValueError('Drive download exceeds limit')
                    chunks.append(chunk)
                return b''.join(chunks),r.headers.get('content-type','')
        raise ValueError('Too many Drive download redirects')
    def file(self,identifier,resource_key=None):
        params={'id':identifier,'export':'download'}
        if resource_key:params['resourcekey']=resource_key
        url='https://drive.google.com/uc'
        for _ in range(2):
            data,mime=self.fetch(url,params)
            if 'text/html' not in mime and not data.lstrip().lower().startswith((b'<!doctype html',b'<html')):return data
            form=DownloadForm();form.feed(data.decode('utf8',errors='replace'))
            if not form.action or not google_download_host(form.action):break
            url,params=form.action,form.fields
        raise ValueError('Google Drive لا يسمح بتنزيل هذا الملف عبر الرابط العام. شاركه مع حساب البوت أو أرسل ZIP.')
    def read(self,url,identifier):
        key=parse_qs(urlparse(url).query).get('resourcekey',[None])[0]
        if '/folders/' in urlparse(url).path:
            params={'resourcekey':key} if key else None
            html,_=self.fetch('https://drive.google.com/drive/folders/'+identifier,params,limit=5*1024*1024)
            text=html.decode('utf8',errors='replace')
            title=re.search(r'<title>(.*?)</title>',text,re.S)
            if title:self.source_name=unescape(title.group(1)).removesuffix(' - Google Drive').strip()
            entries=folder_entries(text,self.settings.max_pages)
        else:
            html,_=self.fetch('https://drive.google.com/file/d/'+identifier+'/view',{'resourcekey':key} if key else None,limit=5*1024*1024)
            class Title(HTMLParser):
                inside=False;value=''
                def handle_starttag(self,t,a):
                    if t=='title':self.inside=True
                def handle_endtag(self,t):
                    if t=='title':self.inside=False
                def handle_data(self,d):
                    if self.inside:self.value+=d
            title=Title();title.feed(html.decode('utf8',errors='replace'))
            name=title.value.removesuffix(' - Google Drive').strip()
            if Path(name).suffix.lower() not in EXTENSIONS|{'.zip'}:raise ValueError('الرابط لا يعرض صورة أو ZIP عامًا قابلًا للتنزيل.')
            self.source_name=Path(name).stem
            entries=[{'id':identifier,'name':name}]
        pages=[];total=0
        for entry in entries:
            new=unpack(entry['name'],self.file(entry['id'],key if entry['id']==identifier else None),self.settings)
            total+=sum(len(data) for _,data in new);pages.extend(new)
            if len(pages)>self.settings.max_pages or total>self.settings.max_batch_bytes:raise ValueError('Drive chapter exceeds limits')
        if not pages:raise ValueError('المجلد لا يحتوي صورًا أو ملفات ZIP عامة مدعومة.')
        return pages
