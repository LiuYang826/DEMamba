import torch
import torch.nn as nn
from layers.Embed import PatchEmbedding
from einops import rearrange, repeat
from layers.DEMamba_Enc import Encoder, EncoderLayer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x): 
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

    
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.enc_in = configs.enc_in
        if self.patch_len == self.stride:
            padding = 0
        else:
            padding = self.stride

        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        self.head_nf = configs.d_model * \
                       int((configs.seq_len // configs.stride))

        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=0)
        

        self.encoder = Encoder(

            [
                EncoderLayer(configs.dropout, d_model = configs.d_model, d_state = configs.d_state,
                               d_conv = configs.d_conv, expand = configs.expand, var=self.enc_in, conv_drop=configs.conv_drop, d_ff=configs.d_ff
                ) for i in range(configs.e_layers)
            ],
            var = self.enc_in,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        B, _, N = x_enc.shape

        enc_out, _ = self.patch_embedding(x_enc.permute(0, 2, 1))

        enc_out = self.encoder(enc_out)

        enc_out = rearrange(enc_out, '(b n) seg_num d_model -> b n d_model seg_num', n = N)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0,2,1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
    

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]

