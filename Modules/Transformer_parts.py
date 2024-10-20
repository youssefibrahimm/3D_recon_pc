import torch
import torch.nn as nn
import torch.nn.functional as F
from sys import modules

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#-------------------------------------------------------------------------------


class Head(nn.Module):
  def __init__(self, n_embed, head_size, dropout, decoder = True, cross_attention = False):
    super(Head, self).__init__()
    # key, query, value matrices (no need for bias)
    self.cross_attention = cross_attention
    self.decoder = decoder
    self.dropout = nn.Dropout(dropout)
    if cross_attention:
      self.keycross = nn.Linear(n_embed, head_size, bias = False)
      self.querycross = nn.Linear(n_embed, head_size, bias = False)
      self.valuecross = nn.Linear(n_embed, head_size, bias = False)
    else:
      self.key = nn.Linear(n_embed, head_size, bias = False)
      self.query = nn.Linear(n_embed, head_size, bias = False)
      self.value = nn.Linear(n_embed, head_size, bias = False)


  def forward(self, x):
    B, T, C = x.shape
    print(f'before k, q, v {x.shape}')
    if self.cross_attention:
      k = self.keycross(x)
      q = self.querycross(x)
      v = self.valuecross(x)
    else:
      k = self.key(x)
      q = self.query(x)
      v = self.value(x)

    # compute attention scores
    wei = (q @ k.transpose(-2, -1) * C**-0.5).to(device) # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
    if self.decoder:
      tril = torch.tril(torch.ones(T, T)).to(device)
      wei = wei.masked_fill(tril == 0, float('-inf')) # makes the past tokens to not communicate with future tokens
    wei = F.softmax(wei, dim = 1) # (B, T, T) the softmax makes the rows sum to 1
    wei = self.dropout(wei)

    # aggregation of the valuse
    out = wei @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
    return out.to(device), k.to(device), v.to(device), q.to(device)
#-------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, dropout, n_embed, decoder = True, cross_attention = False):

    super(MultiHeadAttention, self).__init__()
    self.head_size_Multi = n_embed // num_heads
    self.heads = nn.ModuleList([Head(n_embed, self.head_size_Multi, dropout, decoder = decoder, cross_attention = cross_attention) for _ in range(num_heads)]) # list of heads (B, T, Head_size)
    self.projection = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x,  k_cross=None, q_cross=None, v_cross=None):
    # concatenating them on the head_size dim = (B, T, head_size_Multi * num_heads) which gives the wanted head size (head_size)
    for i, h in enumerate(self.heads):
      out_i = h(x)[0]
      print(f"Shape of head {i} output: {out_i.shape}")
    if k_cross is not None and q_cross is not None and v_cross is not None:
      # Use cross-attention
      out = torch.cat([h(x)[0] for h in self.heads], dim=-1)
      keys = torch.cat([k_cross], dim=-1)
      values = torch.cat([v_cross], dim=-1)
      queries = torch.cat([q_cross], dim=-1)
    else:
      # Use self-attention
      out = torch.cat([h(x)[0] for h in self.heads], dim = -1)
      # get the keys and values to pass them from the encoder to the decoder for cross-attention
      keys = torch.cat([h(x)[1] for h in self.heads], dim = -1)
      values = torch.cat([h(x)[2] for h in self.heads], dim = -1)
      queries = torch.cat([h(x)[3] for h in self.heads], dim = -1)

    # Scales the head_size_Multi * num_heads to n_embed for the ability to call multiple heads with the correct initialization
    out = self.projection(out) # (B, T, n_embed)
    out = self.dropout(out)

    return out.to(device) , keys.to(device), values.to(device), queries.to(device)
#-------------------------------------------------------------------------------
class FeedForward(nn.Module):
  def __init__(self, n_embed, dropout):
    super(FeedForward, self).__init__()
    self.MLP = nn.Sequential(
        nn.Linear(n_embed, 4 * n_embed),
        nn.LeakyReLU(),
        nn.Linear(4 * n_embed, n_embed),
        nn.Dropout(dropout)
    )
  def forward(self, x):
    x = self.MLP(x)
    return x.to(device)
#-------------------------------------------------------------------------------
class encoderBlock(nn.Module):
  def __init__(self, num_heads, n_embed, dropout):
    super(encoderBlock, self).__init__()
    self.multi_head = MultiHeadAttention(num_heads, dropout, n_embed,decoder = False, cross_attention=False)
    self.feed_forward = FeedForward(n_embed, dropout)
    self.LN1 = nn.LayerNorm(n_embed)
    self.LN2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    multi_head_out, k, v, _ =  self.multi_head(self.LN1(x)) # Residual connection
    x = x + multi_head_out
    x = x + self.feed_forward(self.LN2(x))# Residual connection
    return x.to(device), k.to(device), v.to(device)
#-------------------------------------------------------------------------------
class DecoderBlock(nn.Module):
  def __init__(self, num_heads, n_embed, dropout):
    super(DecoderBlock, self).__init__()
    self.multi_head = MultiHeadAttention(num_heads, dropout, n_embed, cross_attention=True, decoder=True) # decoder is initialized to True
    self.feed_forward = FeedForward(n_embed, dropout)
    self.LN1 = nn.LayerNorm(n_embed)
    self.LN2 = nn.LayerNorm(n_embed)
    self.LN3 = nn.LayerNorm(n_embed)

  def forward(self, x, k_encoder, v_encoder):
    # self attention
    self_att_out, _, _,q_decoder  =  self.multi_head(self.LN1(x))
    x = x + self_att_out # Residual connection

    # cross attention: the query(aggregated values) of the decoder and the keys, values of the encoder
    cross_att_out, _, _, _ = self.multi_head(self.LN2(x), k_cross = k_encoder, v_cross = v_encoder, q_cross = q_decoder)
    x = x + cross_att_out # Residual connection

    # feed forward
    ffwd_out = self.feed_forward(self.LN3(x))
    x = x + ffwd_out # Residual connection
    return x.to(device)
#-------------------------------------------------------------------------------
