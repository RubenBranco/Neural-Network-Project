import json
import re
import os
import operator


class ChatParser:

    def __init__(self, file):
        self.file = file

    def parse(self, write_file_name):
        banned_accounts = ['wowsobot', 'twitchnotify', 'moobot', 'hnlbot', 'nightbot', 'alinity_bot', 'xanbot', 'amazquotebot', 'scamazbot', 'zebooom', 'aoaagoldbot', 'baconrobot', 'rept0bot', 'blackbadge', 'bobross', 'boxboxbot', 'carcibot', 'revlobot', 'chinglishtvbot', 'jayridebot', 'cluntbotstovens', 'cloaktato', 'cohhilitionbot', 'waffless_', 'laurieforman', 'jackieburkhart', 'desbot', 'destiny_bot', 'Bot_v2_Beta', 'Logs', 'purrbot', 'docb0t', 'niconicobot', 'supascootpm', 'exbcbot', 'berzekerbot', 'snusbot', 'fourtfbot', 'adeladeexe', 'ambot', 'stahpbot', 'itmebot', 'boterie', 'streamelements', 'kaybotify', 'thebanebot', 'lasskeepobot', 'lasskeepobot', 'loopy_the_merciful', 'lulabot', 'mtgbot', 'mandroid', 'gather_bot', 'onscreenbot', 'spitter_bot', 'proxymybot', 'analyticsbot', 'sheepfarmer', 'rrbot', 'wizebot', 'botmotion', 'startupbot', 'chinnbot', 'mithbot', 'swiftsbot', 'swiftpoints', 'pantsu__bot', 'priestbot', 'x9kbot', 'botseventeen', 'tpp', 'litzbot', 'ohbot', 'hoffmannbot', 'voyscout', 'sc2replaystatsbot', 'wobblerbot', 'jaffamod']
        with open(self.file) as file_reader:
            with open(write_file_name, 'w') as file_writer:
                for line in file_reader:
                    lines = json.loads(line)['text']
                    for chat_line in lines.split('\n'):
                        if len(chat_line) > 0:
                            match = re.findall('([\[\d]*-[\d]*-[\d]* [\d]*:[\d]*:[\d]* UTC\]) ([A-Za-z\d_]*)', chat_line)
                            slicer = len(match[0][0])+len(match[0][1])+3
                            if len(chat_line[slicer:]) > 0:
                                if match[0][1] not in banned_accounts and chat_line[slicer:][0] != '!':
                                    if chat_line[slicer:] != '<Message Deleted>':
                                        chat_line = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', chat_line[slicer:])
                                        chat_line = re.sub('@[A-Za-z\d_]*', '', chat_line)
                                        if chat_line != '':
                                            try:
                                                file_writer.write(chat_line + '\n')
                                            except:
                                                new_line = chat_line.encode('ascii', 'ignore') + b'\n'
                                                file_writer.write(new_line.decode('utf-8'))

    def tokenize(self, log_file, new_log_file_name):
        num_lines = 0 # Might as well get to know the size
        with open(log_file) as file:
            with open(new_log_file_name, 'w') as file_writer:
                for line in file:
                    file_writer.write('<SOS>'+line.strip('\n')+'<EOS>\n')
                    num_lines += 1
        print(num_lines)


if __name__ == '__main__':
    parser = ChatParser(os.path.normpath('C:/Users/Ruben/Desktop/neuralnetwork/Neural-Network-Project/Language Modelling/Scraper/Scraper/twichlogs.jl'))
    parser.parse('twitch.log')
