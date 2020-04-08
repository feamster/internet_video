from flask import Flask, render_template, redirect, request, url_for, session
from youtube_dl import YoutubeDL
import time
from flask_pymongo import PyMongo
import bcrypt

app = Flask(__name__, template_folder='templates')

app.config['MONGO_DBNAME'] = 'admin'
app.config['MONGO_URI'] = 'mongodb://uchivideo1:63czhj6c8pe@18.179.39.139:27017/admin'

mongo = PyMongo(app)

def get_time():
    return str(hash(str(time.time())))

def extract_url(youtube_id):
    ydl = YoutubeDL()
    ydl.add_default_info_extractors()
    weburl = 'https://www.youtube.com/watch?v=' + youtube_id
    info = ydl.extract_info(weburl, download=False)
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
    gen_resolution = set()

    return_video_source = ''

    for x in info['formats']:
        idx = x['format_id']
        if ((idx in audio_list) and first_audio):
            first_audio = False
            return_video_source += "<audio preload id=\"audio\">"
            return_video_source = return_video_source + "<source src=" + quote + x[
                'url'] + quote + ' ' + "type=" + quote + audio_type[idx] + quote + ' ' + "/>" + '\n'
            return_video_source += "</audio>\n"

        if ((idx in resolution) and (resolution[idx] not in gen_resolution)):
            gen_resolution.add(resolution[idx])
            if (not first_video):
                return_video_source = return_video_source + "<source src=" + quote + x[
                    'url'] + quote + ' ' + "type=" + quote + video_type[idx] + quote + ' ' + "label=" + quote + \
                                      resolution[idx] + "p" + quote + ' ' + "res=" + quote + resolution[
                                          idx] + quote + ' ' + "/>"
            else:
                first_video = False
                return_video_source = return_video_source + "<source src=" + quote + x[
                    'url'] + quote + ' ' + "type=" + quote + video_type[idx] + quote + ' ' + "label=" + quote + \
                                      resolution[idx] + "p" + quote + ' ' + "res=" + quote + resolution[
                                          idx] + quote + ' ' + "default label=" + quote + resolution[
                                          idx] + "p" + quote + ' ' + '/>\n'
    return return_video_source


@app.route('/')
def hello_world():
    if 'username' in session:
        return render_template('home.html', userinfo='You are logged in as ' + session['username'] + '.')
    return render_template('home-not-logged-in.html')

@app.route('/home')
def homepage():
    return render_template('video_player.html')

@app.route('/play/<string:page_name>/')
def render_video(page_name):
    video_source = extract_url(page_name)
    return render_template('video_player.html', videosource=video_source)

@app.route('/post_video_survey/')
def post_survey():
    return render_template('post_survey.html')

@app.route('/signup/')
def sign_up():
    return render_template('sign-up-ver2.html')

@app.route('/signuprequest', methods=['POST'])
def signuprequest():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'name' : request.form['username']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())
            users.insert(
                {
                    'name':request.form['username'], 
                    'password': hashpass, 
                    'firstname': request.form['firstname'],
                    'lastname': request.form['lastname'],
                    'email': request.form['email'],
                    'address': request.form['address'],
                    'address2': request.form['address2'],
                    'age': request.form['age'],
                    'device': request.form['device'],
                    'isp': request.form['isp'],
                    'type': request.form['type'],
                    'plan': request.form['plan'],
                    'inlineRadioOptions': request.form['inlineRadioOptions'],
                    'timespent': request.form['timespent'],
                    'resolution': request.form['resolution'],
                    'msize': request.form['msize']
               }
            )
            session['username'] =  request.form['username']
            return redirect(url_for('hello_world'))

        return 'That username already exists!'

    return render_template('sign-up-ver2.html')


@app.route('/loginpage')
def loginpage():
    if 'username' in session:
        return redirect(url_for(hello_world))
    return render_template('login-page.html')
    
@app.route('/login', methods=['POST'])
def login():
    users = mongo.db.users
    login_user = users.find_one({'name': request.form['username']})

    if login_user:
        if bcrypt.hashpw(request.form['password'].encode('utf-8'), login_user['password']) == login_user['password']:
            session['username'] = request.form['username']
            return redirect(url_for('hello_world'))

    return 'Invalid username or password'

@app.route('/post/', methods=['POST'])
def save_data():
    print(request.form)
    return 'Sending sucessfully'


if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.run(debug=True, host='127.0.0.1', port=5000)
