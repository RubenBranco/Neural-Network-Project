import json
import re


class ChatParser:

    def __init__(self, file):
        self.file = file

    def parse(self, write_file_name):
        banned_accounts = ['wowsobot', 'twitchnotify', 'moobot', 'hnlbot', 'nightbot', 'alinity_bot', 'xanbot', 'amazquotebot', 'scamazbot', 'zebooom', 'aoaagoldbot', 'baconrobot', 'rept0bot', 'blackbadge', 'bobross', 'boxboxbot', 'carcibot', 'revlobot', 'chinglishtvbot', 'jayridebot', 'cluntbotstovens', 'cloaktato', 'cohhilitionbot', 'waffless_', 'laurieforman', 'jackieburkhart', 'desbot', 'destiny_bot', 'Bot_v2_Beta', 'Logs', 'purrbot', 'docb0t', 'niconicobot', 'supascootpm', 'exbcbot', 'berzekerbot', 'snusbot', 'fourtfbot', 'adeladeexe', 'ambot', 'stahpbot', 'itmebot', 'boterie', 'streamelements', 'kaybotify', 'thebanebot', 'lasskeepobot', 'lasskeepobot', 'loopy_the_merciful', 'lulabot', 'mtgbot', 'mandroid', 'gather_bot', 'onscreenbot', 'spitter_bot', 'proxymybot', 'analyticsbot', 'sheepfarmer', 'rrbot', 'wizebot', 'botmotion', 'startupbot', 'chinnbot', 'mithbot', 'swiftsbot', 'swiftpoints', 'pantsu__bot', 'priestbot', 'x9kbot', 'botseventeen', 'tpp', 'litzbot', 'ohbot', 'hoffmannbot', 'voyscout', 'sc2replaystatsbot', 'wobblerbot', 'jaffamod']
        with open(self.file) as file_reader:
            with open(write_file_name, 'w') as file_writer:
                lines = json.loads(file_reader.readline())['text']
                for line in lines.split('\n'):
                    if len(line) > 0:
                        match = re.findall('([\[\d]*-[\d]*-[\d]* [\d]*:[\d]*:[\d]* UTC\]) ([A-Za-z\d_]*)', line)
                        slicer = len(match[0][0])+len(match[0][1])+3
                        if match[0][1] not in banned_accounts and line[slicer:][0] != '!':
                            if line[slicer:] != '<Message Deleted>':
                                line = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', line[slicer:])
                                if line != '':
                                    file_writer.write(line + '\n')


if __name__ == '__main__':
    parser = ChatParser('/Volumes/LINUX MINT/twitchchat.jl')
    parser.parse('twitch.log')
