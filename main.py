from tokenization import Tokenizer
from dataloader import TextLoader
from model import Decoder_Stack
from trainer import Trainer
import datetime
import torch
import re

DATA_PATH = r".\data\kaorpus.txt"
TRAIN_TEST_SPLIT_RATIO = 0.95 
BATCH_SIZE = 64
SEQ_LENGTH = 256
EMB_DIM = 384
OUTPUT_LENGTH = 500
SEED = 11
PROMPT = "Welcome to class"
LEARNING_RATE = 3e-4
NUM_ITERATIONS = 100000 # 70,000
WEIGHT_DECAY = 0.01
N_DECODER_LAYERS = 8
N_HEADS = 6
ATTN_DIM = EMB_DIM // N_HEADS
DROPOUT = 0.2
PRINT_EVERY = 500
SAVE_EVERY = 1000
VALIDATE_EVERY = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
CONTINUE_FROM = r'.\checkpoints3\day_2.pt' # Specify File
SAVE_IN = r'.\checkpoints3' # Specify Directory

def generate(inp):
    PROMPT = inp
    torch.manual_seed(SEED)
    device = DEVICE
    tokenizer = Tokenizer(DATA_PATH)
    
    model = Decoder_Stack(
        vocab_size = len(tokenizer),
        emb_dim = EMB_DIM,
        seq_len = SEQ_LENGTH,
        stack_size = N_DECODER_LAYERS,
        n_heads = N_HEADS,
        attn_dim = ATTN_DIM,
        dropout_rate= DROPOUT,
        device= device
        )
    
    model.to(device)
    
    if CONTINUE_FROM is not None:
        model.load_state_dict(torch.load(CONTINUE_FROM))
    else:
        print('WARNING: NO MODEL LOADED')
    
    print(model.generate(PROMPT, OUTPUT_LENGTH, tokenizer, device))

def main():
    torch.manual_seed(SEED)
    device = DEVICE
    tokenizer = Tokenizer(DATA_PATH)
    data = tokenizer.encode(tokenizer.text)
    
    cutoff = int(len(data)*TRAIN_TEST_SPLIT_RATIO)
    train_data, val_data = data[:cutoff], data[cutoff:]
    train_loader = TextLoader(train_data,
                            BATCH_SIZE, 
                            SEQ_LENGTH
                            )
    val_loader = TextLoader(val_data,
                            BATCH_SIZE, 
                            SEQ_LENGTH
                            )

    model = Decoder_Stack(
        vocab_size = len(tokenizer),
        emb_dim = EMB_DIM,
        seq_len = SEQ_LENGTH,
        stack_size = N_DECODER_LAYERS,
        n_heads = N_HEADS,
        attn_dim = ATTN_DIM,
        dropout_rate= DROPOUT,
        device= device
        )
    
    if CONTINUE_FROM is not None:
        model.load_state_dict(torch.load(CONTINUE_FROM))

    trainer = Trainer(device)
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(),
                            LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY
                            )
    trainer.train(train_loader, 
                val_loader, 
                model, 
                NUM_ITERATIONS, 
                loss, 
                optim, 
                SAVE_IN,
                PRINT_EVERY,
                SAVE_EVERY,
                VALIDATE_EVERY
                )
    
    if SAVE_IN is not None:
        now = re.sub(r'[ .:]', '_', str(datetime.datetime.now()))
        torch.save(model.state_dict(), SAVE_IN+"\\"+now+".pt")
    
    print(model.generate(PROMPT, OUTPUT_LENGTH, tokenizer, device))
    pass

if __name__ == "__main__":
    inp = input("Enter prompt | -1 to train\n")
    if inp == "-1":
        main()
    else:
        generate(inp)

# add weight initilization
# add better loss function, visualization of training, tqdm
# add validation loss
# experiment with gelu
# add beam search
# clean up constants at the beginning
# add API to just generate, maybe using different algorithms
