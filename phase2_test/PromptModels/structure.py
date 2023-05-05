import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# class Prompt(nn.Module):
#     def __init__(self, type='Deep', depth=12, channel=768, length=10):
#         super().__init__()
#         self.prompt = nn.Parameter(torch.zeros(depth, length, channel))
#         trunc_normal_
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AdaBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, reduction_dim=32):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        #self.adapter_down = nn.Linear(dim, reduction_dim)
        #self.adapter_act = nn.GELU()
        #self.adapter_up = nn.Linear(reduction_dim, dim)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #adapt = self.adapter_down(x)
        #adapt = self.adapter_act(adapt)
        #adapt = self.adapter_up(adapt)
        #x = x + adapt
        return x

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        #out = x.mm(self.weight)
        return cosine

class PromptLearner(nn.Module):
    def __init__(self, num_classes, prompt_length, prompt_depth, prompt_channels):
        """_summary_

        Args:
            num_classes (int): 最后分类的时候变成几个类
            prompt_length (int): 每个地方插入几个Prompt
            prompt_depth (int): ViT有几层需要插入Prompt
            prompt_channels (int): 表示一个prompt的嵌入维度
        """
        super().__init__()
        # 这些参数被 VPT_ViT 插入
        self.Prompt_Tokens = nn.Parameter(torch.zeros(prompt_depth, prompt_length, prompt_channels)) # 
        # 这个就是cosine classifier
        self.head = NormedLinear(prompt_channels, num_classes)
        trunc_normal_(self.Prompt_Tokens, std=.02) # 使用高斯分布初始化，超出边界的会trunc
        #trunc_normal_(self.head.weight, std=.02)
    def forward(self, x):
        return self.head(x)

class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU, weight_init='', Prompt_Token_num=1, VPT_type="Deep"):

        # Recreate ViT
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                         representation_size, distilled, drop_rate, attn_drop_rate, drop_path_rate, embed_layer,
                         norm_layer, act_layer, weight_init)

        self.VPT_type = VPT_type
        # if VPT_type == "Deep":
        #     self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
        # else:  # "Shallow"
        #     self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))
        # trunc_normal_(self.Prompt_Tokens, std=.02)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            AdaBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.dropout = nn.Dropout(0.1)
        self.prompt_learner = PromptLearner(num_classes, Prompt_Token_num, depth, embed_dim)
        self.head = nn.Identity() # Prompt Learner的 head后面还有一层head

    # def New_CLS_head(self, new_classes=15):
    #     self.head = nn.Linear(self.embed_dim, new_classes)
    #     trunc_normal_(self.head.weight, std=.02)
    
    def Freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

        # self.Prompt_Tokens.requires_grad_(True)
        for param in self.prompt_learner.parameters():
            param.requires_grad_(True)

    def obtain_prompt(self):
        # 导出prompt
        prompt_state_dict = {'head': self.head.state_dict(),
                             'Prompt_Tokens': self.Prompt_Tokens}
        # print(prompt_state_dict)
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):
        # 导入prompt
        self.head.load_state_dict(prompt_state_dict['head'])
        self.Prompt_Tokens = prompt_state_dict['Prompt_Tokens']

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":
            
            # 每一层插入几个Token
            Prompt_Token_num = self.prompt_learner.Prompt_Tokens.shape[1]
            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.prompt_learner.Prompt_Tokens[i].unsqueeze(0)
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1] # x 是 (batch, 196, 16*16*3)
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num] # 为了实现方便，算完之后再取需要的

        else:  # self.VPT_type == "Shallow"
            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            # Sequntially procees
            x = self.blocks(x)

        x = self.norm(x)
        return self.pre_logits(x[:, 0])  # use cls token for cls head

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.prompt_learner(x)
        x = self.head(x)
        return x
