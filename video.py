import requests
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import Any
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import datetime
import re
import pickle
import redis
import arxiv
import asyncio
from mutagen.mp3 import MP3
from moviepy.editor import *
import edge_tts
import codecs
from langchain.vectorstores.redis import Redis
from mutagen.mp3 import MP3
import edge_tts
from langchain.chat_models import ChatOpenAI
from nider.core import Font
from nider.core import Outline
from nider.models import Content, Header, Image
from langchain.llms import OpenAI

girl = "zh-CN-XiaoyiNeural"  
boy = 'zh-CN-YunxiNeural'

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_BASE'] = 'https://api.aiproxy.io/v1'
redis_url = os.getenv('REDIS_URL')
r = redis.from_url(redis_url)

llm = OpenAI(max_tokens=10000,model_name='gpt-3.5-turbo-16k')
embeddings = OpenAIEmbeddings()
path = os.getcwd() + '/'

def get_paper_info(id,max_results=1):

    big_slow_client = arxiv.Client(
        page_size = 1,
        delay_seconds = 10,
        num_retries = 1
    )

    search = arxiv.Search(
        id_list=[id],
        max_results = max_results,
    )

    result = big_slow_client.results(search)
    result = list(result)[0]
    title = result.title
    abstract = result.summary
    return title,abstract.replace('\n',' ').replace('{','').replace('}','')

def get_cover(title):
    header = Header(text=title,
                    text_width=50,
                    font=Font(path=path+'Handwritten-English-2.ttf',size=40),
                    align='center',
                    color='#000100',
                    )
    content = Content(header=header)
    img = Image(content,fullpath='./'+id+'/cover.png')
    img.draw_on_image(path+'cover.jpg')
    os.rename('./'+id+'/cover.png','./'+id+'/cover.jpg')
    with open('./'+id+'/cover.jpg', 'rb') as f:
        file_content = f.read()
    r.set('bilibili:'+id+':radio_cover.jpg',file_content)

def generate_readme(id):
    if not os.path.exists('./'+id):
        os.mkdir('./'+id)
        os.mkdir('./'+id+'/audio')
    title,abstract  = get_paper_info(id)
    get_cover(title)
    prompt_template =  """现在你是Paperweekly论文电台分享主播Shirin，请根据论文标题"%s"和摘要"%s",生成一段电台播报风格的采访稿，严格按照一问一答的形式生成，角色分别是主播Shirin和AI研究人员Ian，生成内容如下：""" %(title,abstract)
    PROMPT = PromptTemplate(template=prompt_template, input_variables=[])
    chain = LLMChain(llm=llm, prompt=PROMPT)
    output = chain.run(c='')
    with open('./'+id+'/r.txt','w') as f:
        f.write(output)
    
async def gen_voice(text,id,idx,voice):
    communicate = edge_tts.Communicate(text, voice)  
    await communicate.save('./'+id+'/audio/'+str(idx)+'.mp3')

def get_time_count(audio_file):
    audio = MP3(audio_file)
    time_count = int(audio.info.length)
    return time_count

def generate_radio(id):
    generate_readme(id)
    a = ''
    with open('./'+id+'/r.txt','r') as f:
        data = f.read()
    data = data.replace('：',':')
    r.set('bilibili:'+id+':r.txt',data)
    output = data.split('\n\n')
    for idx,o in enumerate(output):
        _o = o.split(':')
        if idx == 0:
            a = _o[0]
        if _o[0] == a:
            asyncio.run(gen_voice(_o[1],id,idx,girl))
        else:
            asyncio.run(gen_voice(_o[1],id,idx,boy))

    image_files = [id+'/cover.jpg'] * 10
    audios = os.listdir('./'+id+'/audio/')
    audios.sort(key=lambda x:int(x[:-4]))
    audio_files = ['./'+id+'/audio/' + a for a in audios]

    total_time = 0

    for audio_file in audio_files:
        total_time += get_time_count(audio_file)

    audio_clip = concatenate_audioclips([AudioFileClip(c) for c in audio_files])
    image_clip = ImageSequenceClip(image_files, fps=len(image_files)/total_time)
    audio_clip = concatenate_audioclips([AudioFileClip(c) for c in audio_files])
    audio_clip.write_audiofile('./'+id+'/'+id+'.mp3')
    with open('./'+id+'/'+id+'.mp3', 'rb') as f:
        file_content = f.read()
    r.set('bilibili:'+id+':radio_'+id+".mp3",file_content)
    print('...generate radio done...')
    video_clip = image_clip.set_audio(audio_clip)
    video_clip.write_videofile('./'+id+'/'+id+'.mp4',codec='libx264')
    with open('./'+id+'/'+id+'.mp4', 'rb') as f:
        file_content = f.read()
    r.set('bilibili:'+id+':radio_'+id+".mp4",file_content)
    r.rpush('radio_cached_ids',id)
    print('...generate video done...')

def get_today_list(day=0):
    ids = r.lrange('radio_cached_ids',0,-1)
    ids = [id.decode("utf-8") for id in ids]
    today = (datetime.date.today() - datetime.timedelta(day)).strftime('%Y-%m-%d')
    url = 'https://huggingface.co/papers?date='+today
    x = requests.get(url)
    data = x.text
    regex = re.compile(r'<a href="/papers/(.*?)"')
    papers = re.findall(regex,data)
    paperlist = list(set(papers))
    paperlist = [paper for paper in paperlist if len(paper) == 10]
    arxivids = [paper for paper in paperlist if paper not in ids]
    return arxivids

if __name__ == '__main__':
    # ids = get_today_list()        
    # print(ids)
    ids = ['2311.01615']
    # for id in ids:
    #     r.rpush('radio_paper',id)
    #     try:
    generate_radio(id)
        # except:
            # print('exception')
