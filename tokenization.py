
class Tokenizer:
    def __init__(self, data_path):
        file = open(data_path, "r")
        self.text = file.read()
        file.close()
        self.vocab = self.char()
        self.char2id = dict(zip(self.vocab, range(len(self.vocab))))
        self.id2char = dict(zip(range(len(self.vocab)), self.vocab))
    
    def __len__(self):
        return len(self.vocab)

    def char(self):
        vocab = sorted(list(set(self.text)))
        return vocab
    
    def encode(self, x):
        return [self.char2id[i] for i in x]
    
    def decode(self, x):
        return "".join([self.id2char[i] for i in x])
    
    def decode_tensor(self, x):
        return (self.decode(x.tolist()))
    
    def decode_batch(self, x):
        return [self.decode_tensor(x[i]) for i in range(x.shape[0])]