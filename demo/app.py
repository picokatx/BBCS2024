from flask import Flask, render_template, Response

import config
from mastodon import Mastodon, StreamListener
from bs4 import BeautifulSoup
import threading
import time
# from model_stuff import BubbleController
# bb_controller = BubbleController()

m = Mastodon(
    access_token = config.token,
    api_base_url = config.instance
)
toots = []

class Listener(StreamListener):
    def on_update(self, status):
#        toot_html = status['content']
#        toot_soup = BeautifulSoup(toot_html, 'html.parser')
#        toot_text = toot_soup.find('p').getText(separator=" ")  #toot_soup.get_text(separator=" ")
#        if check_filters(toot_text):
        toots.append(status)

def getfeed():
    return m.stream_public(Listener())

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/feed')
def feed():
    def toot2html(toot):
        a = BeautifulSoup(toot['content'], 'html.parser').find('p')
        if ((a!=None) and (not ("http" in a.getText(separator=" ")))): print(a.getText(separator=" "))
        return f'''
            <li class="toot">
                <div style="max-height: 64px;">
                    <img width="64" height="64" style="float: left; margin-right: 16px;" src="{toot['account']['avatar']}" />
                    <div>
                        <h3>{toot['account']['username']}</h3>
                        <p>{toot['account']['acct']}</p>
                    </div>
                </div>
                {toot['content']}
            </li>
            '''

    def generate():
        while True:
            while toots:
                if (len(toots)!=0):
                    yield('event: toot\n')
                    
                    for line in toot2html(toots.pop(0)).split('\n'):
                        yield(f'data: {line}\n')
                    yield('\n')
            time.sleep(1)
    
    return Response(generate(), content_type='text/event-stream')

if __name__ == "__main__":
    threading.Thread(target=getfeed).start()
    app.run()
