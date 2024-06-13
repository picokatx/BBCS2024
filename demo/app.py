from flask import Flask, render_template, Response, request
import config
from mastodon import Mastodon, StreamListener
from bs4 import BeautifulSoup
import threading
import time
from demo.model import bubble_controller as bb_controller

m = Mastodon(
    access_token = config.token,
    api_base_url = config.instance
)
ai_threshold = 0.1
toots = []
def check_basic_filters(status):
    html = status['content']
    soup = BeautifulSoup(html, 'html.parser')
    try:
        text = soup.find('p').getText(separator=" ")
    except:
        return False
    if soup.find('a'):
        return False

    if status['account']['bot']:
        return False
    if status['sensitive']:
        return False
    if status['language'] != 'en':
        return False
    if 'http://' in text or 'https://' in text:
        return False

    return True

def check_ai_filters(status):
    global ai_threshold
    html = status['content']
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.find('p').getText(separator=" ")
    try:
        bb_controller.post_message(text, status['account']['username'])
    except:
        print("rate limit")
    disp_out = bb_controller.user_display("travers", status['account']['username']);
    if disp_out[0]>ai_threshold:  # example
        return disp_out[1].replace("Travers", "User").replace("travers", "user")
    if '!' in text:  # example
        return 'exclamation point'
    return None

class Listener(StreamListener):
    def on_update(self, status):
        if check_basic_filters(status):
            toots.append(status)

def getfeed():
    return m.stream_public(Listener())

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/feed')
def feed():
    def toot2html(toot, spoiler=None):
        content = toot['content']
        if spoiler:
            content = f'''
                <details>
                    <summary>
                        <h3>WARNING: {spoiler}</h3>
                    </summary>
                    {content}
                </details>
                '''

        return f'''
            <li class="toot">
                <div style="max-height: 64px;">
                    <img width="64" height="64" style="float: left; margin-right: 16px;" src="{toot['account']['avatar']}" />
                    <div>
                        <h3>{toot['account']['username']}</h3>
                        <p>{toot['account']['acct']}</p>
                    </div>
                </div>
                {content}
            </li>
            '''

    def generate():
        while True:
            while toots:
                toot = toots.pop(0)
                warning = check_ai_filters(toot)
                yield('event: toot\n')
                for line in toot2html(toot, spoiler=warning).split('\n'):
                    yield(f'data: {line}\n')
                yield('\n')
            time.sleep(1)

    return Response(generate(), content_type='text/event-stream')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/out', methods=['POST'])
def out():
    global ai_threshold
    ai_threshold = int(request.form.get("threshold"))/100
    match (request.form.get("command_box")):
        case "userinfo": 
            return bb_controller.get_user_info(request.form.get("args_box"))[1]
        case "dispuser": 
            print(request.form.get("args_box").split("###"))
            return bb_controller.user_display(*request.form.get("args_box").split("###"))[1]
        case "disptopic":
            print(request.form.get("args_box").split("###"))
            return bb_controller.topic_display(*request.form.get("args_box").split("###"))[1]
    return 

if __name__ == "__main__":
    threading.Thread(target=getfeed).start()
    app.run()
