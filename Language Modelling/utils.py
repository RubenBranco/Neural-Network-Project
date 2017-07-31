class DataLoad:
    def __init__(self, batch_size, train_file, max_sequences):
        self.batch_size = batch_size
        self.train_file = train_file
        self.currentBatch = batch_size
        self.max_sequences = max_sequences
        self.offset = 0

    def next_batch(self):
        if self.currentBatch < self.max_sequences:
            batch = []
            with open(self.train_file) as file:
                i = self.currentBatch + 1
                file.seek(self.offset, 0)
                for line in file:
                    if i < self.currentBatch + self.batch_size:
                        if i > self.currentBatch:
                            batch.append(line)
                    else:
                        self.offset = file.tell()
                        break
                    i += 1
            self.currentBatch = self.currentBatch + self.batch_size
            return batch
        else:
            return None

    def initial_batch(self):
        batch = []
        with open(self.train_file) as file:
            i = 0
            for line in file:
                if i < self.currentBatch:
                    batch.append(line)
                else:
                    self.offset = file.tell()
                    break
                i += 1
        return batch

    def reset_offset(self):
        self.offset = 0


class DataFormat:
    @property
    def vocab(self):
        return 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!?.:,;-_^~><\\|´`/*+\'=)([]ößü&%$#"€@§¡ºªãñçáàéèóòôÃÑÇÁÁÉÈÓÒÔ 0123456789'

    def max_len(self, train_file):
        max_length = 0
        vocab = self.vocab
        missing_chars = ''
        with open(train_file) as file_reader:
            for line in file_reader:
                if len(line) > max_length:
                    max_length = len(line)
                for char in line:
                    if char not in vocab and char not in missing_chars:
                        missing_chars += char
        with open('maxseqsize.config', 'w') as file:
            file.write(str(max_length))
        return missing_chars

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
    print(data_format.max_len('twitch.log'))
