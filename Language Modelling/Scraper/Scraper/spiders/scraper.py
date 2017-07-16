import scrapy


class TwitchOverRustle(scrapy.Spider):
    name = 'twitch'

    def start_requests(self):
        urls = [
            'https://overrustlelogs.net'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.channel_parser)

    def channel_parser(self, response):
        banned_channels = ['Zuzu', 'Yznb', 'Yoda', 'Weplaywot', 'Versuta', 'Vegeta777', 'Ungespielt', 'Tvbrekan', 'Taketv_hs', 'Taketv', 'Starladder5', 'Starladder1', 'Sharishaxd', 'Sampev', 'Riotgamesturkish', 'Riotgamesbrazil', 'Miken_tv', 'Lemondogspani', 'Kregme', 'Izakooo', 'Helenalive', 'Gronkh', 'Genietfan', 'Garenatw', 'Eloise', 'Elotrix', 'Elotrixlivestream', 'Eclypsiatvlol', 'Egnofficial', 'Dota2ti', 'Doigby', 'Dido_d', 'Devmehdi', 'Dendi', 'Dafran', 'Chatroulettecyrustv', 'Cheatbanned', 'Chapmad', 'Broeki1', 'Bibaboy', 'Beastqt', 'Baudusau420', 'Bailamos', 'Asiangodtonegg3be0', 'Arigameplays']
        if len(response.url[8:].split('/')) > 3:
            yield {
                'text': response.body.decode('utf-8')
            }
        else:
            for href in response.css('a.collection-item::attr(href)').extract():
                if href.split('%')[0][1:] not in banned_channels:
                    if len(response.url[8:].split('/')) < 3:
                        if 'userlogs' not in href and 'broadcaster' not in href and 'subscribers' not in href:
                            yield response.follow(href, callback=self.channel_parser)
                    else:
                        if 'userlogs' not in href and 'broadcaster' not in href and 'subscribers' not in href:
                            yield response.follow(href + '.txt', callback=self.channel_parser)
