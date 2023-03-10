import torch
import torch.nn
import re
import datetime

class Trainer:
    def __init__(self, device):
        self.device = device
        pass

    def train(self, train_loader, model, num_itr, loss_ctr, optim, SAVE_IN):
        model.train()
        model.to(self.device)
        print_every = 10000
        save_every = 2000
        for i in range(1, num_itr+1):
            inputs, targets = train_loader.get_batch() # batch_size x seq_length+1
            inputs  = inputs.to(self.device)
            targets  = targets.to(self.device)
            outputs = model(inputs)
            batch, seq_len, vocab_size = outputs.shape
            
            outputs = outputs.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len, )
             
            loss = loss_ctr(outputs, targets)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % print_every == 0:
                print(f"iteration: {i} | loss: {loss}")
            if i % save_every == 0:
                if SAVE_IN is not None:
                    now = re.sub(r'[ .:]', '_', str(datetime.datetime.now()))
                    torch.save(model.state_dict(), SAVE_IN+"\\"+now+".pt")
            pass