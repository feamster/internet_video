from flask import Flask, render_template, redirect
from youtube_dl import YoutubeDL
import os
import time

app = Flask(__name__)

def get_time():
	return str(hash(str(time.time())))

def extract_url(youtube_id):
    ydl = YoutubeDL()
    ydl.add_default_info_extractors()
    weburl='https://www.youtube.com/watch?v='+youtube_id;
    info = ydl.extract_info(weburl, download=False)
    resolution = { '278': '144',
                   '242': '240',
                   '243': '360',
                   '244': '480',
                   '247': '720',
                   '248': '1080' }
    quote="\""
    first=True

    outputFileName=youtube_id+"-"+get_time()
    outputFileDir="templates/"+outputFileName+".html"

    webpage_pre=open("webpage_pre.html",mode="r")


    outputFile=open(outputFileDir,'w')

    for line in webpage_pre.readlines():
    	print(line, file=outputFile)

    for x in info['formats']:
    	idx=x['format_id']
#    	print(idx)
    	if (idx in resolution):
    		if (not first):
	    		print("<source src="+quote+x['url']+quote, "type='video/webm; codecs=\"vp9, vorbis\"'", 
    			"res="+resolution[idx],"label="+quote+resolution[idx]+"p"+quote, "/>", file=outputFile)
    		else:
    			first=False
	    		print("<source src="+quote+x['url']+quote, "type='video/webm; codecs=\"vp9, vorbis\"'", 
    			"res="+resolution[idx],"label="+quote+resolution[idx]+"p"+quote, 
    			 "default label="+quote+resolution[idx]+"p"+quote, "/>", file=outputFile)

    webpage_suc=open("webpage_suc.html","r")

    for line in webpage_suc.readlines():
    	print(line, file=outputFile)

    return outputFileName


@app.route('/')
def hello_world():
  return 'hello world!'

@app.route('/video/<youtube_id>')
def f(youtube_id):
	target_url=extract_url(youtube_id)
	return redirect("/play/"+target_url)

@app.route('/play/<string:page_name>/')
def render_static(page_name):
    return render_template('%s.html' % page_name)

if __name__ == '__main__':
  app.run()
