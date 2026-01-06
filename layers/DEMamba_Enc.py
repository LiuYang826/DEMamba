import torch
import torch.nn as nn
from layers.Mamba_TVDS import Mamba_TVDS
import torch.nn.functional as F
from einops import rearrange

class Encoder(nn.Module):
    def __init__(self, smd_layers, var):
        super(Encoder, self).__init__()
        self.smd_layers = nn.ModuleList(smd_layers)
        self.var = var

    def forward(self, x):

        for smd_layer in self.smd_layers:
            x = smd_layer(x, self.var)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, dropout, d_model, d_state, d_conv, expand, var, conv_drop, d_ff):
        super(EncoderLayer, self).__init__()
        
        self.smd = Mamba_TVDS(d_model = d_model, d_state = d_state,
                                    d_conv = d_conv, expand = expand, vars=var, conv_drop=conv_drop)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)

        self.conv1 = nn.Conv1d(in_channels=d_model*var, out_channels=d_ff*var, kernel_size=1, groups=var)
        self.conv2 = nn.Conv1d(in_channels=d_ff*var, out_channels=d_model*var, kernel_size=1, groups=var)

        self.activation = F.gelu


    def forward(self, x, var):

        new_x = self.smd(x, var)
        x = x + self.dropout1(new_x)
        y = self.norm1(x.permute(0,2,1)).permute(0,2,1)

        res = y
        y = rearrange(y, "(b n) p d -> b p (n d)", n=var)
        y = self.dropout2(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout3(self.conv2(y).transpose(-1, 1))
        y = rearrange(y, "b p (n d) -> (b n) p d", n=var)

        return self.norm2((res + y).permute(0,2,1)).permute(0,2,1)
