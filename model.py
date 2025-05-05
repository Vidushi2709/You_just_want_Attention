import torch
import torch.nn as nn
import math

class input_embeddings(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.d_model= d_model
        self.vocab = vocab
        self.embedding= nn.Embedding(vocab, d_model)
    
    def forward(self, x):
        return self.embedding(x)* math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len: int, dropout:float)-> None:
        super().__int__()
        self.d_model= d_model
        self.seq_len = seq_len
        self.dropout= dropout

        #create a matrix of (seq_len, d_model)
        pe= torch.zeros(seq_len, d_model)
        #vector of shape seq_len,1
        pos= torch.arange(0, seq_len, dtype= torch.float).unsqueeze(1)
        #formula PE(pos, 2i)= sin (pos/10000^(2i/d_model))
        #PE(pos, 2i+1)= cos(pos/10000 ^ (2i/d_model))
        div_term= torch.exp(torch.arange(0, d_model,2)* -(math.log(10000.0)/d_model))
        pe[:,0::2]= torch.sin(pos*div_term)
        pe[:,1::2]= torch.cos(pos*div_term)
        pe=pe.unsqueeze(0) # [seq_len, d_model] -> [batch_size, seq_len, d_model]
        self.register_buffer('pe', pe) # Store in model (non-learnable), persistent buffer
    
    def forward(self, x):
        x=x+ (self.pe[:,:x.size(1),:]).requires_grad=False()
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps:float= 10 ** -6)-> None:
        super().__init__()
        self.eps=eps
        self.alpha= nn.Parameter(torch.ones(1)) #multiplicative
        self.beta= nn.Parameter(torch.zeroes(1)) #additive-> bias
    
    def forward(self, x):
        mean= x.mean(dim=-1, keepdim=True)
        std= x.std(dim=-1, keepdim=True)
        return self.alpha* (x-mean)/(std + self.eps) + self.beta 
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1)->None:
        super().__init__()
        self.linear_1= nn.Linear(d_model, d_ff)
        self.dropout= nn.Dropout(dropout)
        self.linear_2= nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        #(batch, se _len, d_model)-> (batch, se _len, d_ff) -> (batch, se _len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# we don't explicity write attention block, include it in multi head attention :)



# here we employ h=8 // heads
# for each we use dk= dv= d_model/h= 64. due to the reduced
#dim of each head, total computation cost is similar to single head attention 
#with full dimensionaltiy

class ResidualConnection(nn.Module):
    def __int__(self, features: int, dropout:float)->None:
        super().__init__()
        self.dropout= dropout
        self.norm= LayerNormalization(features)

    def forward(self, x, sublayer): #sublayer -> prev layer
        return x + self.dropout(sublayer(self.norm(x)))

class Multiheadattention(nn.Module):
    def __inti__(self, h, d_model, dropout=float)->None:
        super().__init__()
        self.d_model= d_model
        self.h=h
        assert d_model %h ==0 #assert -> assume it is true always "d model is not divisible by h"
        self.d_k= d_model//h
        self.w_q= nn.Linear(d_model, d_model)
        self.w_k= nn.Linear(d_model, d_model)
        self.w_v= nn.Linear(d_model, d_model)

        self.w_o= nn.Linear(d_model, d_model)
        self.dropout= nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k= query.shape[-1]
        attention_scores= (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is None:
            attention_scores.masked_fill(mask==0, -1e9) # write a very low value where mask ==0
        attention_scores= attention_scores.softmax(dim=-1) #applied softmax
        if dropout is not None:
            attention_scores= dropout(attention_scores)
            return (attention_scores @ value), attention_scores


    def forward(self,q, k,v, mask): #mask hide stuff
        query = self.w_q(q)
        key= self.w_k(k)
        value= self.w_v(v)

        #.view -> splits last dim into h heads of size d_k
        #transpose(1,2) -> Swaps seq_len and h to get (batch, h, se_len, d_k)
        query= query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        #calculate attention
        x, self.attention_scores= Multiheadattention.attention(query, key, value, mask, self.dropout)

        #concatenate all the heads
        #(batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model) 
        # tranpose(1,2)-> swaps 1 dim with another
        x= x.transpose(1,2).contiguos().view(x.shape[0], -1, self.h*self.d_k)


        return self.w_o(x)
        
class Encoderblock(nn.Module):
    def __init__(self, features: int, self_attention_block: Multiheadattention, feedforward: FeedForward, dropout: float)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feedforward= feedforward
        self.residual_connections= nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x= self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask)) #q=k=v=x
        x= self.residual_connections[1](x, self.feedforward)
        return x
    
# we can have n encoder blocks

class Encoder(nn.Module):
    def __init__(self, features:int, layers: nn.ModuleList)->None:
        super().__init__()
        self.layers= layers
        self.norm= LayerNormalization(features)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x= layer(x, mask)
            return self.norm(x);

class Decoderblock(nn.Module):
    def __init__(self, features: int, self_attention_block: Multiheadattention, cross_attention_block: Multiheadattention, feed_forward_block: FeedForward, dropout: float)->None:
        super().__init__()
        self.self_attention_block= self_attention_block
        self.cross_attention_block= cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections= nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, x, encoder_op, src_mask, tgt_mask): # src_mask-> from encoder tgt_mask-> from decoder
        x= self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_op, encoder_op, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

#we can n decoderblocks

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList)->None:
        super().__init__()
        self.layers= layers
        self.norm= LayerNormalization(features)
    
    def forward(self, x, encoder_op, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_op, src_mask, tgt_mask)
        return self.norm(x)

#Linear layer-> Projection layer

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab)->None:
        super().__init__()
        self.proj= nn.Linear(d_model, vocab)
    def forward(self, x)-> None:
        return self.proj(x)

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embeds= input_embeddings, tgt_embded= input_embeddings, src_pos= PositionalEncoding, tgt_pos= PositionalEncoding, project_layer= ProjectionLayer)->None:
        super().__init__()
        self.encoder= encoder
        self.decoder= decoder
        self.src_embeds= src_embeds
        self.tgt_embeds= tgt_embded
        self.src_pos= src_pos
        self.tgt_pos= tgt_pos
        self.project_layer= project_layer
    
    def encode(self, src, src_mask):
        src= self.src_embeds(src)
        src= self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_op: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask= torch.Tensor):
        tgt= self.tgt_embeds(tgt)
        tgt= self.tgt_pos(tgt)
        return self.decode(tgt, encoder_op, src_mask, tgt_mask)

    def project(self, x):
        return self.project_layer(x)

#now combine all blocks in 1 transformer

def build_a_transformer(src_vocab_size: int, tgt_vocab_size: int, src_len: int, tgt_len: int, d_model:int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048)-> Transformer:

    #create the embedding layers
    src_embeds= input_embeddings(d_model, src_vocab_size)
    tgt_embeds= input_embeddings(d_model, tgt_vocab_size)

    #create positional encoding layers
    src_pos= PositionalEncoding(d_model, src_len, dropout)
    tgt_pos= PositionalEncoding(d_model, tgt_len, dropout)

    #encoder blocks
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block= Multiheadattention(d_model, h, dropout)
        FeedForward_block= FeedForward(d_model, d_ff, dropout)
        encoding= Encoderblock(d_model, encoder_self_attention_block, FeedForward_block, dropout)
        encoder_blocks.append(encoding)
    
    #decoder blocks
    decoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block= Multiheadattention(d_model, h, dropout)
        decoder_cross_attention_block= Multiheadattention(d_model, h, dropout)
        FeedForward_block= FeedForward(d_model, d_ff, dropout)
        decoding= Decoderblock(d_model, encoder_self_attention_block, FeedForward_block, dropout)
        decoder_blocks.append(decoding)

    #encoder && decoder
    encoder= Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder= Decoder(d_model, nn.ModuleList(decoder_blocks))

    #Projection layer
    project_la= ProjectionLayer(d_model, tgt_vocab_size)

    #final transformer
    transformer= Transformer(encoder, decoder, src_embeds, tgt_embeds, src_pos, tgt_pos, project_la)

    for p in transformer.parameters(): # loops thr all model's parameters and applies xavier uniform inilialization to all weight matrics (i.e. tensors with dim()>1)
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    
    return transformer
