import torch
from torch import nn

class Decoder_Stack(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len, stack_size, n_heads, attn_dim, dropout_rate):
        super(Decoder_Stack, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.decoder_stack = nn.Sequential(*[Decoder(n_heads, attn_dim, seq_len, emb_dim, dropout_rate)] * stack_size)
        self.fc= nn.Linear(emb_dim, vocab_size)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.decoder_stack(x)
        x = self.fc(x)
        return x
    
    @torch.no_grad()
    def generate(self, prompt, max_len, tokenizer, device):
        self.eval()
        prompt = tokenizer.encode(prompt)
        output = torch.tensor(prompt).to(device)
        prompt = output[-(self.seq_len):]
        for _ in range(max_len):
            logits = self(prompt)
            logits = logits[-1:, :] # take final logits
            probs = self.softmax(logits)

            # next_id = torch.argmax(probs) # Greedy
            next_id = torch.multinomial(probs, 1) # Sampling
            
            output = torch.cat([output, next_id.view(1)], 0)
            prompt = output[-self.seq_len:]
        return tokenizer.decode_tensor(output)

class Decoder(nn.Module):
    def __init__(self, n_heads, attn_dim, seq_len, emb_dim, dropout_rate):
        super().__init__()
        self.multihead_attn = MultiHeaded_Attention(n_heads, attn_dim, seq_len, emb_dim, dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(emb_dim)
        self.layer_norm_2 = nn.LayerNorm(emb_dim)
        self.feed_forward = nn.Sequential(
            # Mulitplying by 4x is from "Attention is All You Need"
            nn.Linear(emb_dim, 4*emb_dim),
            nn.ReLU(),
            nn.Linear(4*emb_dim, emb_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = x + self.multihead_attn(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class MultiHeaded_Attention(nn.Module):
    def __init__(self, n_heads, attn_dim, seq_len, emb_dim, dropout_rate):
        super().__init__()
        self.attn_heads = nn.ModuleList([Attention_Head(attn_dim, seq_len, emb_dim, dropout_rate)]*n_heads)
        self.downsize = nn.Linear(attn_dim * n_heads, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = [attn_head(x) for attn_head in self.attn_heads]
        x = torch.cat(x, dim=-1)
        x = self.downsize(x)
        x = self.dropout(x)
        return x 
      
class Attention_Head(nn.Module):
    def __init__(self, attn_dim, seq_len, emb_dim, dropout_rate):
        super().__init__()
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))

        self.q_fc = nn.Linear(emb_dim, attn_dim)
        self.k_fc = nn.Linear(emb_dim, attn_dim)
        self.v_fc = nn.Linear(emb_dim, attn_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim = -1)    

    def forward(self, x):
        q = self.q_fc(x)
        k = self.k_fc(x)
        v = self.v_fc(x)
        similarity_scores = q @ k.transpose(-2,-1)
        similarity_scores = similarity_scores * k.shape[-1]**-0.5
        similarity_scores = self.softmax(similarity_scores)
        v = similarity_scores @ v
        return v
 
