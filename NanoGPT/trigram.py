import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Sequential

from torch.optim.lr_scheduler import LambdaLR

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 200
learning_rate = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed=384
n_head=6
n_layer=6
dropout=0.2


# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def custom_lr_lambda(step):
    warmup_steps=4000

    factor=(n_embed**-0.5)*min((step+1)**-0.5,step*(warmup_steps**-1.5))
    return (factor/learning_rate)

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embed,head_size,bias=False)
        self.query=nn.Linear(n_embed,head_size,bias=False)
        self.value=nn.Linear(n_embed,head_size,bias=False)

        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))


        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape

        k=self.key(x) # (B,T,head_size)
        q=self.query(x) # (B,T,head_size)
        
        #compute the attention scores("affinity")
        wei=q@k.transpose(-2,-1) * (C**-0.5)  # (B,T,head_size) @ (B,head_size,T) ---> (B,T,T)
        wei=wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))  # (B,T,T)
        wei=F.softmax(wei,dim=-1) # (B,T,T)

        wei=self.dropout(wei)
        #perform the weighted aggregation of the values
        v=self.value(x) #(B,T,head_size)

        out=wei@v # (B,T,T) @ (B,T,head_size) ---> (B,T,head_size)

        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads) ])
        self.proj=nn.Linear(n_embed,n_embed)  # projection layer going back into the residual pathway
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.proj(out)
        out=self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by non-linearity"""

    def __init__(self,n_embed):
        super().__init__()
        self.net=Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed), # projection layer going back into the residual pathway
            nn.Dropout(dropout),
        )
    
    def forward(self,x):
        return self.net(x)



class Block(nn.Module):
    """Transformer Block : Communication followed by computation"""


    def __init__(self,n_embed,n_head):
        # n_embed : embedding dimension , n_head: number of heads we like

        super().__init__()
        head_size=n_embed//n_head
        self.sa=MultiHeadAttention(n_head,head_size)   # communication
        self.ffwd=FeedForward(n_embed) # computation

        self.ln1=nn.LayerNorm(n_embed) #Layer Normalization on the last dimension (channel dimension)
        self.ln2=nn.LayerNorm(n_embed) #Layer Normalization
    

    def forward(self,x):
        x=x+self.sa(self.ln1(x))   # Residual Connection
        x=x+self.ffwd(self.ln2(x)) # Residual Connection
        return x


# super simple bigram model
class TrigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size*vocab_size, n_embed)
        self.position_embedding_table=nn.Embedding(block_size,n_embed)
        self.blocks=nn.Sequential(*[Block(n_embed,n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embed)   # final layer norm
        self.lm_head=nn.Linear(n_embed,vocab_size)

    
    def forward(self, idx, targets=None):

        B,T=idx.shape

        if T<2:
            logits=torch.zeros(B,T,vocab_size,device=idx.device)
        else :
            prev1=idx[:,:-2]   # (B,T-2)
            prev2=idx[:,1:-1]  # (B,T-2)
            pair_idx=prev1*vocab_size+prev2
            tok_emb = self.token_embedding_table(pair_idx) # (B,T-2,n_embed)
            pad=torch.zeros(B,2,n_embed,device=idx.device)
            tok_emb=torch.cat([pad,tok_emb],dim=1)
            pos_emb=self.position_embedding_table(torch.arange(T,device=device)) # (T,C = n_embed)
            x=tok_emb+pos_emb  #(B,T,C)
            x=self.blocks(x)   # (B,T,C)
            x=self.ln_f(x)     #(B,T,C)
            logits=self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            #crop idx to the last block size tokens
            idx_cond=idx[:,-block_size:]

            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = TrigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler=LambdaLR(optimizer,lr_lambda=custom_lr_lambda)
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))