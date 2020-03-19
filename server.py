from flask import Flask, render_template, redirect, request
from youtube_dl import YoutubeDL
import time

app = Flask(__name__)


def get_time():
    return str(hash(str(time.time())))

def extract_url(youtube_id):
    ydl = YoutubeDL()
    ydl.add_default_info_extractors()
    weburl = 'https://www.youtube.com/watch?v=' + youtube_id
    info = ydl.extract_info(weburl, download = False)
    outputFileName = youtube_id + "-" + get_time()
    outputFileDir = "templates/" + outputFileName + ".html"
    webpage_pre = open("webpage_pre.html", mode = "r")
    outputFile = open(outputFileDir, 'w')
    for line in webpage_pre.readlines():
        print(line, file=outputFile)

    resolution = { 
                   '133': '240',
                   '134': '360',
                   '135': '480',
                   '136': '720',
                   '137': '1080',
                   # '18': '360',
                   # '22': '720',
                   '160': '144',
                   '278': '144',
                   '242': '240',
                   '243': '360',
                   '244': '480',
                   '247': '720',
                   '248': '1080'
    }
    video_type = {
      '133': 'video/mp4', '134': 'video/mp4', '135': 'video/mp4', '136': 'video/mp4', 
      '137': 'video/mp4', '18': 'video/mp4', '22': 'video/mp4', 
      '160': 'video/mp4',
      '278': 'video/webm', '242': 'video/webm', '243': 'video/webm', '244': 'video/webm', 
      '247': 'video/webm', '248': 'video/webm'
    }
    audio_list = ['139', '140', '141', '256', '258', '325', '328', '171', '172', '249', '250', '251']
    audio_type = {
      '139': 'audio/m4a', '140': 'audio/m4a', '141': 'audio/m4a', '256': 'audio/m4a', 
      '258': 'audio/m4a', '325': 'audio/m4a', '328': 'audio/m4a', 
      '171': 'audio/webm', '172': 'audio/webm', '249': 'audio/webm', 
      '250': 'audio/webm', '251': 'audio/webm'
    }

    quote = "\""
    first_video = True
    first_audio = True
    gen_resolution=set()
    
    for x in info['formats']:
        idx = x['format_id']

        if ((idx in audio_list) and first_audio):
            first_audio = False
            print("<audio preload id=\"audio\">", file = outputFile)
            print("<source src=" + quote + x['url'] + quote, "type=" + quote + audio_type[idx] + quote, "/>", file = outputFile)
            print("</audio>", file = outputFile)

        if ((idx in resolution) and (resolution[idx] not in gen_resolution)):
            gen_resolution.add(resolution[idx])
            if (not first_video):
                print("<source src=" + quote + x['url'] + quote, "type=" + quote + video_type[idx] + quote, 
                      "res=" + quote + resolution[idx] + quote, "label=" + quote + resolution[idx] + "p" + quote, 
                      "/>", file = outputFile)
            else:
                first_video = False
                print("<source src=" + quote + x['url'] + quote, "type=" + quote + video_type[idx] + quote, 
                    "res=" + quote + resolution[idx] + quote, "label=" + quote + resolution[idx] + "p" + quote, 
                    "default label=" + quote + resolution[idx] + "p" + quote, "/>", file = outputFile)

    webpage_suc = open("webpage_suc.html", "r")
    for line in webpage_suc.readlines():
        print(line, file = outputFile)

    return outputFileName


@app.route('/')
def hello_world():
    return 'Welcome to UChicago Internet Video QoE Project Website!'

@app.route('/video/<youtube_id>')
def f(youtube_id):
    target_url = extract_url(youtube_id)
    return redirect("/play/" + target_url)


@app.route('/play/<string:page_name>/')
def render_static(page_name):
    return render_template('%s.html' % page_name)

@app.route('/play/post/', methods=['POST'])
def save_data():
    print(request.form)
    return 'welcome'

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
    # app.run(debug=True, host='127.0.0.1', port=5000)
