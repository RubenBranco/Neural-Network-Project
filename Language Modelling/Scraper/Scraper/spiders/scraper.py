import scrapy


class TwitchOverRustle(scrapy.Spider):
    name = 'twitch'

    def start_requests(self):
        urls = [
            'https://overrustlelogs.net/Imaqtpie%20chatlog'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.channel_parser)

    def channel_parser(self, response):
        if len(response.url[8:].split('/')) > 3:
            yield {
                'text': response.body.decode('utf-8')
            }
        else:
            for href in response.css('a.collection-item::attr(href)').extract():
                if len(response.url[8:].split('/')) < 3:
                    if 'userlogs' not in href and 'broadcaster' not in href and 'subscribers' not in href:
                        yield response.follow(href, callback=self.channel_parser)
                else:
                    if 'userlogs' not in href and 'broadcaster' not in href and 'subscribers' not in href:
                        yield response.follow(href + '.txt', callback=self.channel_parser)
