import config
from mastodon import Mastodon, StreamListener
from bs4 import BeautifulSoup

m = Mastodon(
    access_token = config.token,
    api_base_url = config.instance
)

class Listener(StreamListener):
    def on_update(self, status):
        toot_html = status['content']
        toot_soup = BeautifulSoup(toot_html, 'html.parser')
        toot_text = toot_soup.find('p').getText(separator=" ")  #toot_soup.get_text(separator=" ")
        print(toot_text)

listener = Listener()

m.stream_public(listener)
