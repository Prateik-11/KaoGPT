import torch
from torch import nn

class Decoder_Stack(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len, stack_size, n_heads, attn_dim, dropout_rate, device):
        super(Decoder_Stack, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        self.device = device

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_embedding = nn.Embedding(seq_len, emb_dim)
        self.decoder_stack = nn.Sequential(*[Decoder(n_heads, attn_dim, seq_len, emb_dim, dropout_rate)] * stack_size)
        self.fc= nn.Linear(emb_dim, vocab_size)
        self.softmax = nn.Softmax(dim = 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.embedding(x) + self.positional_embedding(torch.arange(x.shape[1], device=self.device))
        x = self.decoder_stack(x)
        x = self.fc(x)
        return x
    
    @torch.no_grad()
    def generate(self, prompt, max_len, tokenizer, device):
        self.eval()
        prompt = tokenizer.encode(prompt)
        output = torch.tensor(prompt).to(device)
        output = output[None, :]
        prompt = output[:, -(self.seq_len):]
        for _ in range(max_len):
            logits = self(prompt)
            # print(logits.shape)
            # break
            logits = logits[:, -1, :] # take final logits
            probs = self.softmax(logits)

            # next_id = torch.argmax(probs) # Greedy
            next_id = torch.multinomial(probs, 1) # Sampling
            
            output = torch.cat([output, next_id], -1)
            prompt = output[-self.seq_len:]
        return tokenizer.decode_batch(output)[0]

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
        T = x.shape[1]
        # print(x.shape)
        # print(q.shape)
        # print(k.shape)
        # print(similarity_scores.shape)
        # print(self.tril.shape)
        # quit()
        similarity_scores = similarity_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        similarity_scores = self.softmax(similarity_scores)
        similarity_scores = self.dropout(similarity_scores)
        v = similarity_scores @ v
        return v
 
