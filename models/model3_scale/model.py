import torch, torch.nn as nn
from einops import rearrange
from torchvision.models import vgg19, VGG19_Weights


def init_params(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    if classname.find('BatchNorm2d') != -1:
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


class FrozenFeatureExtractor(nn.Module):
    
    def __init__(self):
        super(FrozenFeatureExtractor, self).__init__()
        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.model.requires_grad_(False)
    
    def forward(self, x):
        return self.model(x)


class ChannelRMSNorm(nn.Module):
    
    def __init__(self, dim):
        super(ChannelRMSNorm, self).__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))
    
    def forward(self, x):
        normed = nn.functional.normalize(x, dim=1)
        return normed * self.scale * self.gamma


class CrossAttention(nn.Module):
    
    def __init__(self, dim, dim_context, dim_head=64, num_heads=8):
        super(CrossAttention, self).__init__()
        self.norm = ChannelRMSNorm(dim)
        self.norm_context = ChannelRMSNorm(dim_context)
        self.num_heads = num_heads
        self.att_scale = dim_head ** -0.5
        dim_embed = dim_head * num_heads
        self.to_q = nn.Conv2d(dim, dim_embed, 1, bias=False)
        self.to_k = nn.Conv2d(dim_context, dim_embed, 1, bias=False)
        self.to_v = nn.Conv2d(dim_context, dim_embed, 1, bias=False)
        self.proj = nn.Conv2d(dim_embed, dim, 1, bias=False)
    
    def forward(self, x, context):
        x = self.norm(x)
        context = self.norm(context)
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q = rearrange(q, 'b (c n) h w -> (b n) (h w) c', n=self.num_heads)
        k = rearrange(k, 'b (c n) h w -> (b n) (h w) c', n=self.num_heads)
        v = rearrange(v, 'b (c n) h w -> (b n) (h w) c', n=self.num_heads)
        qk = torch.einsum('b m c, b n c -> b m n', q, k) * self.att_scale
        attn = torch.einsum('b m n, b n c -> b m c', qk.softmax(dim=-1) + 1e-8, v) + 1e-8
        attn = rearrange(attn, '(b n) (h w) c -> b (c n) h w', n=self.num_heads, h=x.size(2))
        return self.proj(attn)


class MutualCrossAttentionBlock(nn.Module):
    
    def __init__(self, in1_dim, in2_dim, out_dim, dim_head=64, num_heads=8, projection=False, projection_dim=64):
        super(MutualCrossAttentionBlock, self).__init__()
        self.projection = projection
        self.block1 = CrossAttention(in1_dim, in2_dim, dim_head, num_heads)
        self.block2 = CrossAttention(in2_dim, in1_dim, dim_head, num_heads)
        if projection:
            self.proj1 = nn.Conv2d(in1_dim, projection_dim, 1, 1, 0, bias=False)
            self.proj2 = nn.Conv2d(in2_dim, projection_dim, 1, 1, 0, bias=False)
        concat_dim = 2 * projection_dim if projection else in1_dim + in2_dim
        self.output = nn.Conv2d(concat_dim, out_dim, 1, 1, 0, bias=False)
    
    def forward(self, x1, x2):
        y1 = self.block1(x1, x2)
        y2 = self.block2(x2, x1)
        if self.projection:
            y1 = self.proj1(y1)
            y2 = self.proj2(y2)
        yc = torch.cat((y1, y2), dim=1)
        return self.output(yc)


class ScalePredictor(nn.Module):
    
    def __init__(self, embeds_dim=512, latent_dim=64, num_classes=30):
        super(ScalePredictor, self).__init__()
        self.frozen_features = FrozenFeatureExtractor()
        self.mutual_cross_attention1 = MutualCrossAttentionBlock(512, 512, embeds_dim)
        self.mutual_cross_attention2 = MutualCrossAttentionBlock(512, 512, embeds_dim)
        self.mutual_cross_attention3 = MutualCrossAttentionBlock(512, 512, embeds_dim)
        self.downsample = nn.Sequential(
            nn.Conv2d(embeds_dim*3, embeds_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(embeds_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(4, 4)),
            nn.Flatten()
        )
        self.fc_pose_embeds = nn.Sequential(nn.Linear(num_classes, embeds_dim), nn.ReLU(inplace=True))
        self.fc_proj_encode = nn.Sequential(nn.Linear(embeds_dim*17, embeds_dim), nn.ReLU(inplace=True))
        self.fc_deform = nn.Sequential(nn.Linear(2, embeds_dim), nn.ReLU(inplace=True), nn.Linear(embeds_dim, embeds_dim), nn.ReLU(inplace=True))
        self.fc_mu = nn.Linear(2*embeds_dim, latent_dim)
        self.fc_var = nn.Linear(2*embeds_dim, latent_dim)
        self.fc_latent = nn.Sequential(nn.Linear(latent_dim, embeds_dim), nn.ReLU(inplace=True), nn.Linear(embeds_dim, embeds_dim), nn.ReLU(inplace=True))
        self.fc_proj_decode = nn.Sequential(nn.Linear(embeds_dim, embeds_dim), nn.ReLU(inplace=True))
        self.fc_output = nn.Sequential(nn.Linear(2*embeds_dim, embeds_dim), nn.ReLU(inplace=True), nn.Linear(embeds_dim, 2))
        self.init_params()
    
    def init_params(self):
        for module in self.children():
            if module.__class__.__name__ == 'FrozenFeatureExtractor':
                continue
            else:
                module.apply(init_params)
    
    def estimate_shared_features(self, img, img_patch1, img_patch2, seg, seg_patch1, seg_patch2, pose_embedding):
        i1 = self.frozen_features(img)
        i2 = self.frozen_features(img_patch1)
        i3 = self.frozen_features(img_patch2)
        s1 = self.frozen_features(seg)
        s2 = self.frozen_features(seg_patch1)
        s3 = self.frozen_features(seg_patch2)
        f1 = self.mutual_cross_attention1(i1, s1)
        f2 = self.mutual_cross_attention2(i2, s2)
        f3 = self.mutual_cross_attention3(i3, s3)
        fc = torch.cat((f1, f2, f3), dim=1)
        fc = self.downsample(fc)
        em = self.fc_pose_embeds(pose_embedding)
        concat_features = torch.cat((fc, em), dim=-1)
        return self.fc_proj_encode(concat_features)
    
    def encode(self, shared_features, bbox_params):
        deform_features = self.fc_deform(bbox_params)
        concat_features = torch.cat((shared_features, deform_features), dim=1)
        return self.fc_mu(concat_features), self.fc_var(concat_features)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, shared_features, z):
        shared_features_proj = self.fc_proj_decode(shared_features)
        latent_features = self.fc_latent(z)
        concat_features = torch.cat((shared_features_proj, latent_features), dim=1)
        return self.fc_output(concat_features)
    
    def forward(self, img, img_patch1, img_patch2, seg, seg_patch1, seg_patch2, pose_embedding, bbox_params):
        shared_features = self.estimate_shared_features(img, img_patch1, img_patch2, seg, seg_patch1, seg_patch2, pose_embedding)
        mu, logvar = self.encode(shared_features, bbox_params)
        z = self.reparameterize(mu, logvar)
        return self.decode(shared_features, z), mu, logvar
    
    @torch.no_grad()
    def predict(self, img, img_patch1, img_patch2, seg, seg_patch1, seg_patch2, pose_embedding, z):
        shared_features = self.estimate_shared_features(img, img_patch1, img_patch2, seg, seg_patch1, seg_patch2, pose_embedding)
        return self.decode(shared_features, z)

