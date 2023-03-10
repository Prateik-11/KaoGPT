import torch

class TextLoader:
    def __init__(self, data, batch_size, seq_length):
        self.batch_size = batch_size
        self.data = data
        self.seq_length = seq_length
    
    def get_batch(self):
        random_indices = torch.randint(
            low = 0,
            high = len(self.data)-(self.seq_length+1),
            size = (self.batch_size,)
            )
        sequences = [torch.tensor(self.data[i:i+self.seq_length]) for i in random_indices]
        random_indices += 1
        targets = [torch.tensor(self.data[i:i+self.seq_length]) for i in random_indices]
        return torch.stack(sequences), torch.stack(targets)

