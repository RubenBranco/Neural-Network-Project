import smtplib
import codecs
import time
from datetime import timedelta


class DataLoad:
    def __init__(self, batch_size, train_file, max_sequences):
        self.batch_size = batch_size
        self.train_file = train_file
        self.max_sequences = max_sequences
        self.offset = 0
        self.batch_num = 0
        self.vocabulary = self.vocab
        self.prohibited_chars = self.prohib_chars

    @property
    def vocab(self):
        return 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!?.:,;-_^~><\\|´`/*+\'«»=÷)({}[]ößü&%$#"€£@§¡¿ºªãñçâáàâéèêóòôõúùíìîýỳÃÑÇÂÂÁÀÉÈÊÓÒÔÍÌÚÙÝỲÕÎ 0123456789'

    @property
    def prohib_chars(self):
        return 'ößüµäÅÖð¯Ë½¥ÄØÆÞÐ¬­¤²ëïÜ©×·Ï¢«Ì¶±ÿ³¦þ¼¹¾Ù®'

    def next_batch(self):
        if self.batch_num * self.batch_size <= self.max_sequences:
            batch = []
            with open(self.train_file) as file:
                file.seek(self.offset, 0)
                for line in file:
                    if len(batch) <= self.batch_size:
                        if self.line_check(line):
                            batch.append(line)
                    else:
                        self.offset = file.tell()
                        break
            return batch
        else:
            return None

    def initial_batch(self):
        batch = []
        with open(self.train_file) as file:
            for line in file:
                if len(batch) <= self.batch_size:
                    if self.line_check(line):
                        batch.append(line)
                else:
                    self.offset = file.tell()
                    break
        self.batch_num += 1
        return batch

    def line_check(self, string):
        for char in self.prohibited_chars:
            if char in string:
                return False
        return True

    def reset_offset(self):
        self.offset = 0


class DataFormat:
    @property
    def vocab(self):
        return 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!?.:,;-_^~><\\|´`/*+\'«»=÷)({}[]ößü&%$#"€£@§¡¿ºªãñçâáàâéèêóòôõúùíìîýỳÃÑÇÂÂÁÀÉÈÊÓÒÔÍÌÚÙÝỲÕÎ 0123456789'

    def max_len(self, train_file):
        max_length = 0
        vocab = self.vocab
        missing_chars = ''
        with codecs.open(train_file, encoding='latin-1') as file_reader:
            file_reader.seek(0, 2)
            size = file_reader.tell()
            file_reader.seek(0, 0)
            percentage = 0
            start_timer = time.time()
            for line in file_reader:
                if len(line) > max_length:
                    max_length = len(line)
                for char in line:
                    if char not in vocab and char not in missing_chars:
                        missing_chars += char
                if int((int(file_reader.tell())/int(size))*100) > percentage:
                    end_timer = time.time()
                    percentage = int((int(file_reader.tell())/int(size))*100)
                    print('Current percentage: ' + str(percentage) + '%' + ' - ETA: ' + str(timedelta(seconds=(end_timer - start_timer)*(100-percentage))), flush=True)
                    start_timer = time.time()
        with open('maxseqsize.config', 'w') as file:
            file.write(str(max_length))
        return missing_chars, max_length

    def one_hot(self, string):
        vocab = self.vocab
        matrix = []
        for char in string:
            matrice = []
            for vocab_char in vocab:
                matrice.append((1 if vocab_char == char else 0))
            matrix.append(matrice)
        return matrix

    @staticmethod
    def seq_len(file_name):
        seq_len = 0
        with open(file_name) as file:
            seq_len = int(file.read())
        return seq_len

if __name__ == '__main__':
    data_format = DataFormat()
    chars = data_format.max_len('/home/ruben/PycharmProjects/Ruben/twitch.log')
    print(chars[0])
    print(chars[1])
