import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Mamba_TVDS(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        vars=7,
        conv_drop=0.1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)


        self.conv1d_pw = nn.Conv1d(
            in_channels=self.d_inner*vars,
            out_channels=self.d_inner*vars,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner*vars,
            padding=d_conv//2,
            **factory_kwargs,
        )


        self.activation = "silu"
        self.act = nn.SiLU()
        
        self.x_proj_PN = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_PN = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.x_proj_PN_i = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_PN_i = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_PN.weight, dt_init_std)
            nn.init.constant_(self.dt_proj_PN_i.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_PN.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_proj_PN_i.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt_PN = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        dt_PN_i = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt_PN = dt_PN + torch.log(-torch.expm1(-dt_PN))
        inv_dt_PN_i = dt_PN_i + torch.log(-torch.expm1(-dt_PN_i))
        with torch.no_grad():
            self.dt_proj_PN.bias.copy_(inv_dt_PN)
            self.dt_proj_PN_i.bias.copy_(inv_dt_PN_i)

        self.dt_proj_PN.bias._no_reinit = True
        self.dt_proj_PN_i.bias._no_reinit = True

        A_PN = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_PN_log = torch.log(A_PN) 
        self.A_PN_log = nn.Parameter(A_PN_log)
        self.A_PN_log._no_weight_decay = True

        A_PN_i = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_PN_log_i = torch.log(A_PN_i)
        self.A_PN_log_i = nn.Parameter(A_PN_log_i)
        self.A_PN_log_i._no_weight_decay = True

        self.D_PN = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_PN._no_weight_decay = True
        self.D_PN_i = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_PN_i._no_weight_decay = True

        self.output = nn.Linear(self.d_inner, self.d_model)
        self.bn = nn.BatchNorm1d(self.d_inner*vars)
        self.drop_pw = nn.Dropout(conv_drop)
        init_weights = torch.ones(2)
        self.weights = nn.Parameter(init_weights)

    def forward(self, hidden_states, inver, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        ssm_state = None
        # conv_state, ssm_state = None, None
        # if inference_params is not None:
        #     conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
        #     if inference_params.seqlen_offset > 0:
        #         out, _, _ = self.step(hidden_states, conv_state, ssm_state)
        #         return out

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A_PN = -torch.exp(self.A_PN_log.float())  
        A_PN_i = -torch.exp(self.A_PN_log_i.float())  
        

        x, z = xz.chunk(2, dim=1)


        x_pw = rearrange(x, "(b n) d p -> b (n d) p", n=inver)
        x_pw = self.conv1d_pw(x_pw)
        x_pw = self.bn(x_pw)
        x_pw = self.act(x_pw)
        x_pw = self.drop_pw(x_pw)

        
        pn = rearrange(x_pw, "b (n d) p -> (b p) d n", n=inver)


        x_PN = pn

        x_PN_i = pn.flip([-1])
        

        z_PN = rearrange(z, '(b n) d p -> (b p) d n', n=inver)

        z_PN_i = rearrange(z, '(b n) d p -> (b p) d n', n=inver)
        z_PN_i = z_PN_i.flip([-1])
        


        x_dbl_PN = self.x_proj_PN(rearrange(x_PN, "b d l -> (b l) d"))  
        dt_PN, B_PN, C_PN = torch.split(x_dbl_PN, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_PN = self.dt_proj_PN.weight @ dt_PN.t()
        dt_PN = rearrange(dt_PN, "d (b l) -> b d l", l=inver)
        B_PN = rearrange(B_PN, "(b l) dstate -> b dstate l", l=inver).contiguous()
        C_PN = rearrange(C_PN, "(b l) dstate -> b dstate l", l=inver).contiguous()
        assert self.activation in ["silu", "swish"]
        y_PN = selective_scan_fn(
            x_PN,
            dt_PN,
            A_PN,
            B_PN,
            C_PN,
            self.D_PN.float(),
            z=z_PN,
            delta_bias=self.dt_proj_PN.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )

        x_dbl_PN_i = self.x_proj_PN_i(rearrange(x_PN_i, "b d l -> (b l) d"))  
        dt_PN_i, B_PN_i, C_PN_i = torch.split(x_dbl_PN_i, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_PN_i = self.dt_proj_PN_i.weight @ dt_PN_i.t()
        dt_PN_i = rearrange(dt_PN_i, "d (b l) -> b d l", l=inver)
        B_PN_i = rearrange(B_PN_i, "(b l) dstate -> b dstate l", l=inver).contiguous()
        C_PN_i = rearrange(C_PN_i, "(b l) dstate -> b dstate l", l=inver).contiguous()
        assert self.activation in ["silu", "swish"]
        y_PN_i = selective_scan_fn(
            x_PN_i,
            dt_PN_i,
            A_PN_i,
            B_PN_i,
            C_PN_i,
            self.D_PN_i.float(),
            z=z_PN_i,
            delta_bias=self.dt_proj_PN_i.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )

        # if ssm_state is not None:
        #     y, last_state = y
        #     ssm_state.copy_(last_state)


        y_PN = rearrange(y_PN, "(b p) d n -> (b n) p d", p=seqlen)
        y_PN_i = y_PN_i.flip([-1])
        y_PN_i = rearrange(y_PN_i, '(b p) d n -> (b n) p d', p=seqlen)

        norm_weights = F.softmax(self.weights, dim=0)
        output = norm_weights[0]*y_PN+norm_weights[1]*y_PN_i
        
        out = self.output(output)

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
