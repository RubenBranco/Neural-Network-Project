class DataLoad:
    def __init__(self, batch_size, train_file, max_sequences):
        self.batch_size = batch_size
        self.train_file = train_file
        self.currentBatch = batch_size
        self.max_sequences = max_sequences
        batch = []
        with open(train_file) as file:
            i = 0
            for line in file:
                if i < self.currentBatch:
                    batch.append(line)
                else:
                    self.offset = file.tell()
                    break
                i += 1

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


class DataFormat:
    @staticmethod
    def max_len(train_file):
        max_length = 0
        vocab = {'<EOS>': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9, 'I': 10, 'J': 11, 'K': 12, 'L': 13,
         'M': 14
            , 'N': 15, 'O': 16, 'P': 17, 'Q': 18, 'R': 19, 'S': 20, 'T': 21, 'U': 22, 'V': 23, 'W': 24, 'X': 25,
         'Y': 26, 'Z': 27,
         'a': 28, 'b': 29, 'c': 30, 'd': 31, 'e': 32, 'f': 33, 'g': 34, 'h': 35, 'i': 36, 'j': 37, 'k': 38, 'l': 39,
         'm': 40,
         'n': 41, 'o': 42, 'p': 43, 'q': 44, 'r': 45, 's': 46, 't': 47, 'u': 48, 'v': 49, 'w': 50, 'x': 51, 'y': 52,
         'z': 53, '!': 54,
         '?': 55, ':': 56, ',': 57, ';': 58, '-': 59, '_': 60, '^': 61, '~': 62, '\\': 63, '|': 64, '´': 65, '`': 66,
         '/': 67, '*': 68,
         '+': 69, "'": 70, '=': 71, ')': 72, '(': 73, '&': 74, '%': 75, '$': 76, '#': 77, '"': 78, '§': 79, '€': 80,
         'º': 81, 'ª': 82,
         'ã': 83, 'ñ': 84, 'ç': 85, 'á': 86, 'à': 87, 'é': 88, 'è': 89, 'ó': 90, 'ò': 91, 'ô': 92, 'Ã': 93, 'Ñ': 94,
         'Ç': 95, 'Á': 96,
         'À': 97, 'É': 98, 'È': 99, 'Ó': 100, 'Ò': 101, 'Ô': 102, ' ':103}
        with open(train_file) as file_reader:
            for line in file_reader:
                if len(line) > max_length:
                    max_length = len(line)
                for char in line:
                    if char not in vocab:
                        print(char)
        with open('maxseqsize.config', 'w') as file:
            file.write(str(max_length))
        return max_length
