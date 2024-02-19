import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from PIL import Image
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from typing import Sequence, Tuple, Type, Union
from monai.utils import optional_import
rearrange, _ = optional_import("einops", name="rearrange")


class FocalUNETR(nn.Module):
    def __init__(
        self,
        img_size: int = 224, 
        patch_size = 2,
        feature_size = 48,
        in_channels: int = 4,    
        out_channels: int = 8,
        window_size = 7, 
        depths: Sequence[int] = (2, 2, 2, 2), 
        num_heads: Sequence[int] = (3, 6, 12, 24),
        drop_rate: float = 0.1,
        dropout_path_rate: float = 0.2,
        normalize: bool = False,
        focal_levels=[2,2,2,2],
        expand_sizes=[3,3,3,3],
        expand_layer="all",
        focal_windows=[7,5,3,1],
        use_conv_embed=True,  
        use_shift=False, 
        spatial_dims = 2, 
        norm_name: Union[Tuple, str] = "instance",
        
    ) -> None:
        
        super().__init__()
        
        self.normalize = normalize
        
        self.FocalViT = FocalTransformer(
            img_size=img_size, in_chans=in_channels, patch_size=patch_size, 
            embed_dim=feature_size, depths=depths, drop=drop_rate, drop_path_rate=dropout_path_rate, 
            focal_levels=focal_levels, expand_sizes=expand_sizes, expand_layer=expand_layer, 
            num_heads=num_heads, focal_windows=focal_windows, window_size=window_size,
            use_conv_embed=use_conv_embed, use_shift=use_shift)
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )  
        
    
    def forward(self, x_in):
        hidden_states_out = self.FocalViT(x_in, normalize=self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_partition_noreshape(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows_h, num_windows_w, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def get_topk_closest_indice(q_windows, k_windows, topk=1):
    coords_h_q = torch.arange(q_windows[0])
    coords_w_q = torch.arange(q_windows[1])
    
    if q_windows[0] != k_windows[0]:
        factor = k_windows[0] // q_windows[0]
        coords_h_q = coords_h_q * factor + factor // 2
        coords_w_q = coords_w_q * factor + factor // 2
    else:
        factor = 1 
    
    coords_q = torch.stack(torch.meshgrid([coords_h_q, coords_w_q]))

    coords_h_k = torch.arange(k_windows[0])
    coords_w_k = torch.arange(k_windows[1])
    coords_k = torch.stack(torch.meshgrid([coords_h_k, coords_w_k]))

    coords_flatten_q = torch.flatten(coords_q, 1)  
    coords_flatten_k = torch.flatten(coords_k, 1)  

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  
    
    relative_position_dists = torch.sqrt(relative_coords[0].float()**2 + relative_coords[1].float()**2)

    topk = min(topk, relative_position_dists.shape[1])
    topk_score_k, topk_index_k = torch.topk(-relative_position_dists, topk, dim=1) 
    indice_topk = topk_index_k
    relative_coord_topk = torch.gather(relative_coords, 2, indice_topk.unsqueeze(0).repeat(2, 1, 1))
    return indice_topk, relative_coord_topk.permute(1, 2, 0).contiguous().float(), topk

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, input_resolution, expand_size, shift_size, window_size, window_size_glo, focal_window, 
                    focal_level, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pool_method="none", topK=64):

        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.window_size = window_size  
        self.window_size_glo = window_size_glo
        self.pool_method = pool_method
        self.input_resolution = input_resolution 
        self.num_heads = num_heads
        head_dim = dim // num_heads        
        self.scale = qk_scale or head_dim ** -0.5
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.nWh, self.nWw = self.input_resolution[0] // self.window_size[0], self.input_resolution[1] // self.window_size[1]
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.topK = topK
 
        coords_h_window = torch.arange(self.window_size[0]) - self.window_size[0] // 2
        coords_w_window = torch.arange(self.window_size[1]) - self.window_size[1] // 2        
        coords_window = torch.stack(torch.meshgrid([coords_h_window, coords_w_window]), dim=-1)  
        self.register_buffer("window_coords", coords_window)

        self.coord2rpb_all = nn.ModuleList()

        self.topks = []
        for k in range(self.focal_level):
            if k == 0:
                range_h = self.input_resolution[0]
                range_w = self.input_resolution[1]
            else:
                range_h = self.nWh
                range_w = self.nWw
            
            topk_closest_indice, topk_closest_coord, topK_updated = get_topk_closest_indice(
                (self.nWh, self.nWw), (range_h, range_w), self.topK)
            self.topks.append(topK_updated)

            if k > 0:
                topk_closest_coord = topk_closest_coord * self.window_size[0]
            topk_closest_coord_window = topk_closest_coord.unsqueeze(1) + coords_window.view(-1, 2)[None, :, None, :]

            self.register_buffer("topk_cloest_indice_{}".format(k), topk_closest_indice)
            self.register_buffer("topk_cloest_coords_{}".format(k), topk_closest_coord_window)

            coord2rpb = nn.Sequential(
                nn.Linear(2, head_dim), 
                nn.ReLU(inplace=True),
                nn.Linear(head_dim, self.num_heads)
            )
            self.coord2rpb_all.append(coord2rpb)

    def forward(self, x_all, mask_all=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x_all[0] 

        B, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, nH, nW, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  
        q_windows = window_partition(q, self.window_size[0]).view(
            -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
            ).transpose(1, 2)

        k_all = []; v_all = []; topKs = []; topk_rpbs = []        
        for l_k in range(self.focal_level):        
            topk_closest_indice = getattr(self, "topk_cloest_indice_{}".format(l_k))
            topk_indice_k = topk_closest_indice.view(1, -1).repeat(B, 1)

            topk_coords_k = getattr(self, "topk_cloest_coords_{}".format(l_k))
            window_coords = getattr(self, "window_coords")

            topk_rpb_k = self.coord2rpb_all[l_k](topk_coords_k)
            topk_rpbs.append(topk_rpb_k)
            
            if l_k == 0:
                k_k = k.view(B, -1, self.num_heads, C // self.num_heads)
                v_k = v.view(B, -1, self.num_heads, C // self.num_heads)
            else:
                x_k = x_all[l_k]
                qkv_k = self.qkv(x_k).view(B, -1, 3, self.num_heads, C // self.num_heads)
                k_k, v_k = qkv_k[:,:,1], qkv_k[:,:,2]

            k_k_selected = torch.gather(k_k, 1, topk_indice_k.view(B, -1, 1).unsqueeze(-1).repeat(1, 1, self.num_heads, C // self.num_heads))
            v_k_selected = torch.gather(v_k, 1, topk_indice_k.view(B, -1, 1).unsqueeze(-1).repeat(1, 1, self.num_heads, C // self.num_heads))
            
            k_k_selected = k_k_selected.view((B,) + topk_closest_indice.shape + (self.num_heads, C // self.num_heads,)).transpose(2, 3)
            v_k_selected = v_k_selected.view((B,) + topk_closest_indice.shape + (self.num_heads, C // self.num_heads,)).transpose(2, 3)

            k_all.append(k_k_selected.view(-1, self.num_heads, topk_closest_indice.shape[1], C // self.num_heads))
            v_all.append(v_k_selected.view(-1, self.num_heads, topk_closest_indice.shape[1], C // self.num_heads))
            topKs.append(topk_closest_indice.shape[1])
                
        k_all = torch.cat(k_all, 2)
        v_all = torch.cat(v_all, 2)
        
        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        attn = (q_windows @ k_all.transpose(-2, -1)) 
        window_area = self.window_size[0] * self.window_size[1]        
        window_area_whole = k_all.shape[2]

        topk_rpb_cat = torch.cat(topk_rpbs, 2).permute(0, 3, 1, 2).contiguous().unsqueeze(0).repeat(B, 1, 1, 1, 1).view(attn.shape)
        attn = attn + topk_rpb_cat

        attn = self.softmax(attn)        
        attn = self.attn_drop(attn)
        
        x = (attn @ v_all).transpose(1, 2).flatten(2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N, window_size, unfold_size):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        for k in range(self.focal_level):
            flops += self.num_heads * N * (self.dim // self.num_heads) * self.topks[k]    
            if k == 0:
                Nq = N
            else:
                window_size_glo = math.floor(self.window_size[0] / (2 ** (k-1)))
                Nq = N // (window_size_glo**2)
            flops += Nq * self.topks[k] * (2 * (self.dim // self.num_heads) + (self.dim // self.num_heads) * self.num_heads)
        for k in range(self.focal_level):
            flops += self.num_heads * N * (self.dim // self.num_heads) * self.topks[k]    
   
        flops += N * self.dim * self.dim
        return flops

class FocalTransformerBlock(nn.Module):
    r""" Focal Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="none",  
                 focal_level=1, focal_window=1, topK=64, use_layerscale=False, layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.use_layerscale = use_layerscale

        if min(self.input_resolution) <= self.window_size:
            self.expand_size = 0
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.window_size_glo = self.window_size

        self.pool_layers = nn.ModuleList()
        if self.pool_method != "none":
            for k in range(self.focal_level-1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                if self.pool_method == "fc":
                    self.pool_layers.append(nn.Linear(window_size_glo * window_size_glo, 1))
                    self.pool_layers[-1].weight.data.fill_(1./(window_size_glo * window_size_glo))
                    self.pool_layers[-1].bias.data.fill_(0)
                elif self.pool_method == "conv":
                    self.pool_layers.append(nn.Conv2d(dim, dim, kernel_size=window_size_glo, stride=window_size_glo, groups=dim))

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, input_resolution=input_resolution, expand_size=self.expand_size, shift_size=self.shift_size, window_size=to_2tuple(self.window_size), 
            window_size_glo=to_2tuple(self.window_size_glo), focal_window=focal_window, 
            focal_level=self.focal_level, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            pool_method=pool_method, topK=topK)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): shape B, C, H, W

        Returns:
            out: B, C, H, W
        """
        x_shape = x.shape
        B0, C0, H0, W0 = x.shape
        x = x.view(B0, C0, H0*W0).transpose(1,2).contiguous() 
        
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size = [self.window_size, self.window_size, self.window_size]
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size = [self.window_size, self.window_size]
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]
        
        

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        x_windows_all = [shifted_x]
        x_window_masks_all = [self.attn_mask]
        
        if self.focal_level > 1 and self.pool_method != "none": 
            for k in range(self.focal_level-1):     
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                pooled_h = math.ceil(H / self.window_size) * (2 ** k)
                pooled_w = math.ceil(W / self.window_size) * (2 ** k)
                H_pool = pooled_h * window_size_glo
                W_pool = pooled_w * window_size_glo

                x_level_k = shifted_x
                if H > H_pool:
                    trim_t = (H - H_pool) // 2
                    trim_b = H - H_pool - trim_t
                    x_level_k = x_level_k[:, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = (H_pool - H) // 2
                    pad_b = H_pool - H - pad_t
                    x_level_k = F.pad(x_level_k, (0,0,0,0,pad_t,pad_b))
                
                if W > W_pool:
                    trim_l = (W - W_pool) // 2
                    trim_r = W - W_pool - trim_l
                    x_level_k = x_level_k[:, :, trim_l:-trim_r]
                elif W < W_pool:
                    pad_l = (W_pool - W) // 2
                    pad_r = W_pool - W - pad_l
                    x_level_k = F.pad(x_level_k, (0,0,pad_l,pad_r))

                x_windows_noreshape = window_partition_noreshape(x_level_k.contiguous(), window_size_glo)  
                nWh, nWw = x_windows_noreshape.shape[1:3]
                if self.pool_method == "mean":
                    x_windows_pooled = x_windows_noreshape.mean([3, 4]) 
                elif self.pool_method == "max":
                    x_windows_pooled = x_windows_noreshape.max(-2)[0].max(-2)[0].view(B, nWh, nWw, C)                  
                elif self.pool_method == "fc":
                    x_windows_noreshape = x_windows_noreshape.view(B, nWh, nWw, window_size_glo*window_size_glo, C).transpose(3, 4) 
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).flatten(-2)                     
                elif self.pool_method == "conv":
                    x_windows_noreshape = x_windows_noreshape.view(-1, window_size_glo, window_size_glo, C).permute(0, 3, 1, 2).contiguous() 
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).view(B, nWh, nWw, C)            

                x_windows_all += [x_windows_pooled]
                x_window_masks_all += [None]
        
        attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all) 

        attn_windows = attn_windows[:, :self.window_size ** 2]
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x if (not self.use_layerscale) else (self.gamma_1 * x))
        x = x + self.drop_path(self.mlp(self.norm2(x)) if (not self.use_layerscale) else (self.gamma_2 * self.mlp(self.norm2(x))))

        x = x.transpose(1,2).contiguous().view(B0, C0, H0, W0)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        img_size (tuple[int]): Resolution of input feature.
        in_chans (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, img_size, patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False, norm_layer=nn.LayerNorm, use_pre_norm=False, is_stem=False):
        super().__init__()
        self.input_resolution = img_size
        self.dim = in_chans
        self.reduction = nn.Linear(4 * in_chans, 2 * in_chans, bias=False)
        self.norm = norm_layer(4 * in_chans)

    def forward(self, x):
        """
        x: B, C, H, W
        """       
        B, C, H, W = x.shape 

        x = x.permute(0, 2, 3, 1).contiguous()

        x0 = x[:, 0::2, 0::2, :]  
        x1 = x[:, 1::2, 0::2, :]  
        x2 = x[:, 0::2, 1::2, :] 
        x3 = x[:, 1::2, 1::2, :] 
        x = torch.cat([x0, x1, x2, x3], -1) 
        x = x.view(B, -1, 4 * C)  

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, expand_size, expand_layer,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, pool_method="none", 
                 focal_level=1, focal_window=1, topK=64, use_conv_embed=False, use_shift=False, use_pre_norm=False, 
                 downsample=None, use_checkpoint=False, use_layerscale=False, layerscale_value=1e-4):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if expand_layer == "even":
            expand_factor = 0
        elif expand_layer == "odd":
            expand_factor = 1
        elif expand_layer == "all":
            expand_factor = -1
        else:
            print('please set expan_factor')
            expand_factor = 0
        # build blocks
        self.blocks = nn.ModuleList([
            FocalTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0 if (i % 2 == 0) else window_size // 2) if use_shift else 0,
                                 expand_size=0 if (i % 2 == expand_factor) else expand_size, 
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, 
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pool_method=pool_method, 
                                 focal_level=focal_level, 
                                 focal_window=focal_window, 
                                 topK=topK, 
                                 use_layerscale=use_layerscale, 
                                 layerscale_value=layerscale_value)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, patch_size=2, in_chans=dim, embed_dim=2*dim, 
                use_conv_embed=use_conv_embed, norm_layer=norm_layer, use_pre_norm=use_pre_norm, 
                is_stem=False
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
            B_new, L_new, C_new = x.shape
            HW_new = int(math.sqrt(L_new)) 
            x = x.transpose(1,2).contiguous().view(B_new, C_new, HW_new, HW_new)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class FocalTransformer(nn.Module):
    r""" Focal Transformer: Focal Self-attention for Local-Global Interactions in Vision Transformer

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
        use_shift (bool): Whether to use window shift proposed by Swin Transformer. We observe that using shift or not does not make difference to our Focal Transformer. Default: False
        focal_stages (list): Which stages to perform focal attention. Default: [0, 1, 2, 3], means all stages 
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level. Default: [1, 1, 1, 1] 
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1] 
        expand_stages (list): Which stages to expand the finest grain window. Default: [0, 1, 2, 3], means all stages 
        expand_sizes (list): The expand size for the finest grain level. Default: [3, 3, 3, 3] 
        expand_layer (str): Which layers we want to expand the window for the finest grain leve. This can save computational and memory cost without the loss of performance. Default: "all" 
        use_conv_embed (bool): Whether use convolutional embedding. We noted that using convolutional embedding usually improve the performance, but we do not use it by default. Default: False 
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value for layer scale. Default: 1e-4 
        use_pre_norm (bool): Whether use pre-norm in patch merging/embedding layer to control the feature magtigute. Default: False
    """
    def __init__(self, 
                img_size=224, 
                patch_size=2, 
                in_chans=3, 
                num_classes=1000,
                embed_dim=96, 
                depths=[2, 2, 6, 2], 
                num_heads=[3, 6, 12, 24],
                window_size=7, 
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, 
                ape=False, 
                patch_norm=True,
                use_checkpoint=False,                 
                use_shift=False, 
                focal_stages=[0, 1, 2, 3], 
                focal_levels=[1, 1, 1, 1], 
                focal_windows=[7, 5, 3, 1], 
                focal_topK=64, 
                focal_pool="fc", 
                expand_stages=[0, 1, 2, 3], 
                expand_sizes=[3, 3, 3, 3],
                expand_layer="all", 
                use_conv_embed=False, 
                use_layerscale=False, 
                layerscale_value=1e-4, 
                use_pre_norm=False, 
                **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            spatial_dims=2,
        )
        patch_size_2D = to_2tuple(patch_size)
        img_size_2D = to_2tuple(img_size)
        patches_resolution = [img_size_2D[0] // patch_size_2D[0], img_size_2D[1] // patch_size_2D[1]]
        num_patches = patches_resolution[0] * patches_resolution[1]
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, 
                               qk_scale=qk_scale,
                               drop=drop_rate, 
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer, 
                               pool_method=focal_pool if i_layer in focal_stages else "none",
                               downsample=PatchMerging,
                               focal_level=focal_levels[i_layer], 
                               focal_window=focal_windows[i_layer], 
                               topK=focal_topK, 
                               expand_size=expand_sizes[i_layer], 
                               expand_layer=expand_layer,                           
                               use_conv_embed=use_conv_embed,
                               use_shift=use_shift, 
                               use_pre_norm=use_pre_norm, 
                               use_checkpoint=use_checkpoint, 
                               use_layerscale=use_layerscale, 
                               layerscale_value=layerscale_value)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x
    
    def forward(self, x, normalize=False):
        x0 = self.patch_embed(x)
        if self.ape:
            x0 = x0 + self.absolute_pos_embed
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        
        assert len(self.layers) == 4
        x1 = self.layers[0](x0)
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers[1](x1)
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers[2](x2)
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers[3](x3)
        x4_out = self.proj_out(x4, normalize)
        
        return [x0_out, x1_out, x2_out, x3_out, x4_out]