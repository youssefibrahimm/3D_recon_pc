from model.model import TDR

# same initliazations as in Transformer_parts 
block_size = 32
dropout = 0.4
n_embed = 128
num_heads = 2
num_encoder_blocks = 2
num_decoder_blocks = 2
 
model = TDR(n_embed, 231666, 4096,256, num_heads, 500000)

parameters = [p.numel() for p in model.parameters()]
print(sum(parameters))
