import torch
import os
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import random



class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(160,608), in_channels=3, embed_dim=768, patch_size=16):
        super(PatchEmbedding, self).__init__()
        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):

        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        # print(B, N, C)
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
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


class Channel_Attention(nn.Module):
    def __init__(self, in_channel = 768, ratio=4):
        super(Channel_Attention, self).__init__()

        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):

        max_pool, _ = torch.max(inputs, dim=1)
        avg_pool = torch.mean(inputs, dim=1)


        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)


        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)


        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)


        x = x_maxpool + x_avgpool


        x = self.sigmoid(x)

        x = x.unsqueeze(1)

        

        return x

class CABlock(nn.Module):
    def __init__(self):
        super(CABlock, self).__init__()

        self.ca = Channel_Attention()

    def forward(self, x):
        
        x_weight = self.ca(x)
        output = x * x_weight
        return output

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(Block, self).__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=hidden_features, act_layer=act_layer, drop=drop)
        self.caBlock = CABlock()

    def forward(self, x):
        
        
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x_CA = x[:, 1:].clone()
        x_CA = self.caBlock(x_CA)
        x[:, 1:] += x_CA
        
        return x

        




class VisionTransformer(nn.Module):
    def __init__(self, img_size=(160, 608), patch_size=16, in_chans=64, num_values=6, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                  norm_layer=nn.LayerNorm, dropout=0.):
        super(VisionTransformer, self).__init__()

        self.patch_embed_64ch = PatchEmbedding(img_size=img_size, in_channels=64, embed_dim=embed_dim,
                                              patch_size=patch_size)

        num_patches = self.patch_embed_64ch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads = num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                  act_layer=nn.GELU, norm_layer=norm_layer) for _ in range(12)
        ])

        self.norm = nn.LayerNorm(embed_dim)


        self.head = nn.Linear(embed_dim, num_values)  # N+1 inputs, regression output


    def forward(self, flow):
        batch_size = flow.size(0)
        x = self.patch_embed_64ch(flow)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 0]

        output = self.head(x)

        return output



class VOT(nn.Module):
    def __init__(self, img_size=(160, 608), patch_size=16, in_chans=64, num_values=6, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1,
                  norm_layer=nn.LayerNorm, dropout=0.1):
        super(VOT, self).__init__()
        self.model = VisionTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_values=num_values, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                  norm_layer=norm_layer, dropout=dropout)



    def forward(self, flow):
        x = self.model(flow)
        return x

def main():
    model = VOT()
    print(model)
if __name__ == "__main__":

    main()

