import torch
import torch.nn as nn
from NAF_net import NAFBlock
from restormer_arch import TransformerBlock
from torch.nn.utils.spectral_norm import spectral_norm
import functools
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn  import functional as F

class _CBINorm(_BatchNorm):
    def __init__(self, num_features, num_con=8, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super(_CBINorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.ConBias = nn.Sequential(
            nn.Linear(num_con, num_features),
            nn.Tanh()
        )
        
    def _check_input_dim(self, input):
        raise NotImplementedError
        
    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        version = metadata.get('version', None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ('running_mean', 'running_var'):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    'Unexpected running stats buffer(s) {names} for {klass} '
                    'with track_running_stats=False. If state_dict is a '
                    'checkpoint saved before 0.4.0, this may be expected '
                    'because {klass} does not track running stats by default '
                    'since 0.4.0. Please remove these keys from state_dict. If '
                    'the running stats are actually needed, instead set '
                    'track_running_stats=True in {klass} to enable them. See '
                    'the documentation of {klass} for details.'
                    .format(names=" and ".join('"{}"'.format(k) for k in running_stats_keys),
                            klass=self.__class__.__name__))
                for key in running_stats_keys:
                    state_dict.pop(key)

        super(_CBINorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
    
    def forward(self, input, ConInfor):
        # print(input.shape)
        # print(ConInfor.shape)
        self._check_input_dim(input)
        b, c = input.size(0), input.size(1)
    
        # Check if ConInfor has batch dimension, add if missing
        if ConInfor.dim() == 1:
            # If ConInfor is a 1D tensor (missing batch dimension)
            ConInfor = ConInfor.unsqueeze(0).expand(b, -1)
    
        tarBias = self.ConBias(ConInfor).view(b, c, 1, 1)
    
        out = F.instance_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats, self.momentum, self.eps)
    
        if self.affine:
            bias = self.bias.repeat(b).view(b, c, 1, 1)
            weight = self.weight.repeat(b).view(b, c, 1, 1)
            return (out.view(b, c, *input.size()[2:]) + tarBias) * weight + bias
        else:
            return out.view(b, c, *input.size()[2:]) + tarBias
        

class CBINorm2d(_CBINorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        
def get_norm_layer(layer_type='instance', num_con=2):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        # c_norm_layer = functools.partial(CBBNorm2d, affine=True, num_con=num_con)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(CBINorm2d, affine=True, num_con=num_con)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer, c_norm_layer

def get_nl_layer(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'sigmoid':
        nl_layer = nn.Sigmoid
    elif layer_type == 'tanh':
        nl_layer = nn.Tanh
    else:
        raise NotImplementedError('nl_layer layer [%s] is not found' % layer_type)
    return nl_layer  
        

class MultiTransformerBlock(nn.Module):
    def __init__(self, num_blocks, channels, num_heads, ffn_expansion_factor, bias):
        super(MultiTransformerBlock, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=channels,
                             num_heads=num_heads,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class RestormerWithNAFBlocks(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 base_channels=32,
                 out_channels=3,
                 dim=256,
                 num_blocks=[2,2,4,8,5],
                 heads=[1,2,4,8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 emb_dim = 256,
                 num_con = 7,
                 dual_pixel_task=False):
        super(RestormerWithNAFBlocks, self).__init__()

        self.dim = dim
        self.num_blocks = num_blocks
        self.heads = heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.bias = bias
        self.LayerNorm_type = LayerNorm_type
        self.emb_dim = emb_dim

        norm_layer, c_norm_layer = get_norm_layer(layer_type='instance', num_con=num_con)
        nl_layer = get_nl_layer(layer_type='relu')




        # Encoder levels
        self.proj = nn.Conv2d(inp_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prompt_conv0 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.n0 = c_norm_layer(base_channels)
        self.a0 = nl_layer()

        self.encoder_level1 = self.make_transformer_block(base_channels,num_blocks[0],num_heads=2)
        self.cross_up1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.task_conv1 = TaskAdaptiveConv2D(base_channels,base_channels,kernel_size=3,stride=1,padding = 1,task_vector_size = emb_dim)
        self.prompt_conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.n1 = c_norm_layer(base_channels)
        self.a1 = nl_layer()
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 2, 2)  # downsample by factor of 2

        self.encoder_level2 = self.make_transformer_block(base_channels*2,num_blocks[1])
        self.cross_up2 = nn.ConvTranspose2d(base_channels*2,base_channels*2,kernel_size=2,stride=2,padding=0,output_padding=0)
        # self.task_conv2 = TaskAdaptiveConv2D(base_channels*2,base_channels*2,kernel_size=3,stride=1,padding = 1,task_vector_size = emb_dim)
        self.prompt_conv2 = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.n2 = c_norm_layer(base_channels*2)
        self.a2 = nl_layer()
        self.down2 = nn.Conv2d(base_channels*2, base_channels*4, 2, 2)

        self.encoder_level3 = self.make_transformer_block(base_channels*4,num_blocks[2],num_heads=8)
        self.prompt_conv3 = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, stride=1, padding=1, bias=bias)
        self.n3 = c_norm_layer(base_channels*4)
        self.a3 = nl_layer()
        self.task_conv3 = TaskAdaptiveConv2D(base_channels*4,base_channels*4,kernel_size=3,stride=1,padding = 1,task_vector_size = emb_dim)
        self.down3 = nn.Conv2d(base_channels*4, base_channels*8, 2, 2)

        # Latent
        self.latent = self.make_transformer_block(base_channels*8,num_blocks[3],num_heads=8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.reduce_chan_level3 = nn.Conv2d(int(base_channels*2**3), int(base_channels*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = self.make_transformer_block(base_channels*4,num_blocks[3],num_heads=8)
        # self.task_conv4 = TaskAdaptiveConv2D(base_channels*4,base_channels*4,kernel_size=3,stride=1,padding = 1,task_vector_size = emb_dim)
        

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.reduce_chan_level2 = nn.Conv2d(int(base_channels*2**2), int(base_channels*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = self.make_transformer_block(base_channels*2,num_blocks[2])
        self.cross_up3 = nn.ConvTranspose2d(base_channels*2,base_channels*2,kernel_size=2,stride=2,padding=0,output_padding=0)
        self.task_conv5 = TaskAdaptiveConv2D(base_channels*2,base_channels*2,kernel_size=3,stride=1,padding = 1,task_vector_size = emb_dim)

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.reduce_chan_level1 = nn.Conv2d(int(base_channels*2), int(base_channels), kernel_size=1, bias=bias)
        self.decoder_level1 = self.make_transformer_block(base_channels, num_blocks[1])
        self.cross_up4 = nn.Conv2d(base_channels,base_channels,kernel_size=3,stride=1,padding=1,bias=bias)
        self.task_conv6 = TaskAdaptiveConv2D(base_channels,base_channels,kernel_size=3,stride=1,padding = 1,task_vector_size = emb_dim)

        # NAFBlocks
        self.nafblock1 = NAFBlock(base_channels)
        self.prompt_conv4 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.n4 = c_norm_layer(base_channels)
        self.a4 = nl_layer()
        self.task_conv7 = TaskAdaptiveConv2D(base_channels,base_channels,kernel_size=3,stride=1,padding = 1,task_vector_size = emb_dim)
        self.proj2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.nafblock2 = NAFBlock(2*base_channels)
        self.prompt_conv5 = nn.Conv2d(2*base_channels, 2*base_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.n5 = c_norm_layer(2*base_channels)
        self.a5 = nl_layer()
        self.task_conv8 = TaskAdaptiveConv2D(base_channels*2,base_channels*2,kernel_size=3,stride=1,padding = 1,task_vector_size = emb_dim)
        self.proj3 = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, stride=1, padding=1, bias=bias)

        # self.nafblock3 = NAFBlock(4*base_channels)
        # self.proj4 = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, stride=1, padding=1, bias=bias)

        # self.nafblock4 = NAFBlock(4*base_channels)
        # self.proj5 = nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, stride=1, padding=1, bias=bias)

        self.nafblock7 = NAFBlock(2*base_channels)
        self.prompt_conv6 = nn.Conv2d(2*base_channels, 2*base_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.n6 = c_norm_layer(2*base_channels)
        self.a6 = nl_layer()
        self.task_conv9 = TaskAdaptiveConv2D(base_channels*2,base_channels*2,kernel_size=3,stride=1,padding = 1,task_vector_size = emb_dim)
        self.proj8 = nn.Conv2d(base_channels*2, base_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.nafblock8 = NAFBlock(base_channels)
        self.prompt_conv7 = nn.Conv2d(base_channels,base_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.n7 = c_norm_layer(base_channels)
        self.a7 = nl_layer()
        self.task_conv10 = TaskAdaptiveConv2D(base_channels,base_channels,kernel_size=3,stride=1,padding = 1,task_vector_size = emb_dim)

        self.proj_final = nn.Conv2d(base_channels, out_channels, kernel_size=3,stride=1,padding=1)
        self.refinement = self.make_transformer_block(out_channels,num_blocks[4],num_heads=1)
        self.prompt_conv8 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.n8 = c_norm_layer(out_channels)
        self.a8 = nl_layer()
        self.task_conv11 = TaskAdaptiveConv2D(out_channels,out_channels,kernel_size=3,stride=1,padding = 1,task_vector_size = emb_dim)

        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=1)

        self.tanh = nn.Tanh()

    def make_transformer_block(self, channels,num_blocks,num_heads=4):
        return MultiTransformerBlock(num_blocks=num_blocks,
                                     channels=channels,
                                     num_heads=num_heads,
                                     ffn_expansion_factor=self.ffn_expansion_factor,
                                     bias=self.bias)


    def forward(self, x,emb, task_vector):
        
        x1 = self.proj(x)
        x1 = self.a0(self.n0(self.prompt_conv0(x1),task_vector))


        
        # Encoder

        out_enc_level1 = self.encoder_level1(x1)          # B,C,H,W
        cross_up_1 = self.cross_up1(out_enc_level1)
        out_enc_level1 = self.a1(self.n1(self.prompt_conv1(out_enc_level1),task_vector))
        # out_enc_level1 = self.task_conv1(out_enc_level1,emb)

        out_enc_level2_in = self.down1(out_enc_level1)     # B,2C,H/2,W/2
        out_enc_level2 = self.encoder_level2(out_enc_level2_in)
        cross_up_2 = self.cross_up2(out_enc_level2)
        out_enc_level2= self.a2(self.n2(self.prompt_conv2(out_enc_level2),task_vector))
        # out_enc_level2 = self.task_conv2(out_enc_level2,emb)

        out_enc_level3_in = self.down2(out_enc_level2)     # B,4C,H/4,W/4
        out_enc_level3 = self.encoder_level3(out_enc_level3_in)
        out_enc_level3 = self.task_conv3(out_enc_level3,emb)
        out_enc_level3 = self.a3(self.n3(self.prompt_conv3(out_enc_level3),task_vector))
        
        out_enc_level4_in = self.down3(out_enc_level3)     # B,8C,H/8,W/8

        
        # Latent
        latent = self.latent(out_enc_level4_in)
        
        # Decoder
        out_dec_level3_in = self.up3(latent) 
        out_dec_level3_in = torch.cat([out_dec_level3_in, out_enc_level3], 1)
        out_dec_level3_in = self.reduce_chan_level3(out_dec_level3_in)
        out_dec_level3 = self.decoder_level3(out_dec_level3_in)
        # out_dec_level3 = self.task_conv4(out_dec_level3,emb)
        
        out_dec_level2_in = self.up2(out_dec_level3) 
        out_dec_level2_in = torch.cat([out_dec_level2_in, out_enc_level2], 1)
        out_dec_level2_in = self.reduce_chan_level2(out_dec_level2_in)
        out_dec_level2 = self.decoder_level2(out_dec_level2_in)
        cross_up_3 = self.cross_up3(out_dec_level2)
        out_dec_level2 = self.task_conv5(out_dec_level2,emb)
        
        out_dec_level1_in = self.up1(out_dec_level2)

        
        out_dec_level1_in = torch.cat([out_dec_level1_in, out_enc_level1], 1)
        out_dec_level1_in = self.reduce_chan_level1(out_dec_level1_in)
        out_dec_level1 = self.decoder_level1(out_dec_level1_in)
        cross_up_4 = self.cross_up4(out_dec_level1)
        out_dec_level1 = self.task_conv6(out_dec_level1,emb)

        # NAF-Stream
        naf_out1 = self.nafblock1(x1 + cross_up_1)
        naf_out1 = self.a4(self.n4(self.prompt_conv4(naf_out1),task_vector))
        naf_out1 = self.task_conv7(naf_out1,emb)
        naf_in2 = self.proj2(naf_out1)

        naf_out2 = self.nafblock2(naf_in2 + cross_up_2)
        naf_out2 = self.a5(self.n5(self.prompt_conv5(naf_out2),task_vector))
        naf_out2 = self.task_conv8(naf_out2,emb)
        naf_in7 = self.proj3(naf_out2)

        # print("done3")
        naf_out7 = self.nafblock7(naf_in7 + cross_up_3)
        naf_out7 = self.a6(self.n6(self.prompt_conv6(naf_out7),task_vector))
        naf_out7 = self.task_conv9(naf_out7,emb)
        naf_in8 = self.proj8(naf_out7)

        naf_out8 = self.nafblock8(naf_in8 + cross_up_4)
        naf_out8 = self.a7(self.n7(self.prompt_conv7(naf_out8),task_vector))
        naf_out8 = self.task_conv10(naf_out8,emb)

        # print("done1")
        output = self.proj_final(out_dec_level1 + naf_out8)
        # print("done2")
        output = self.refinement(output)
        output = self.a8(self.n8(self.prompt_conv8(output),task_vector))
        output = self.task_conv11(output,emb)
        output = self.out_conv(output)
        
        output = self.tanh(output)
        # print("done")
        return output

class Conv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, pad_type='reflect', bias=True, norm_layer=None, nl_layer=None):
        super(Conv2dBlock, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        self.conv = spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=0, bias=bias))
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x
        
        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x
                     
    def forward(self, x):
        return self.activation(self.norm(self.conv(self.pad(x))))
    


#### Discriminator

class D_NET(nn.Module):
    def __init__(self, input_nc=3, ndf=32, block_num=6,  norm_type='instance'):
        super(D_NET, self).__init__()
        nl_layer = nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
        block = [Conv2dBlock(input_nc, ndf, kernel_size=4,stride=2,padding=1,bias=False,nl_layer=nl_layer)]
        dim_in=ndf
        for n in range(1, block_num):
            dim_out = min(dim_in*2, ndf*8)
            block += [Conv2dBlock(dim_in, dim_out, kernel_size=4, stride=2, padding=1,bias=False,nl_layer=nl_layer)]
            dim_in = dim_out
        dim_out = min(dim_in*2, ndf*8)
        block += [Conv2dBlock(dim_in, 1, kernel_size=4, stride=1, padding=1,bias=True) ]
        self.conv = nn.Sequential(*block)
        
    def forward(self, x):
        return self.conv(x)
    
import torch
import torch.nn as nn
import torch.nn.functional as F



class TaskAdaptiveConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,bias=True, task_vector_size= 256, hidden_dim=128, multiplier=0.01):
        super(TaskAdaptiveConv2D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.multiplier = multiplier
        self.stride = stride
        self.padding = padding
        
        # Initialize base weights and bias
        self.base_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # self.base_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # Dimensions for task feature map
        
        self.feature_height = kernel_size
        self.feature_width = kernel_size
        self.feature_channels = 4
        
        task_feature_dim = self.feature_channels * self.feature_height * self.feature_width * self.out_channels
        
        # Task vector processing layers
        self.task_fc = nn.Sequential(
            nn.Linear(task_vector_size, task_feature_dim),
            nn.ReLU()
        )
        
        # Transpose convolution to double feature map size
        self.conv1 = nn.Conv2d(
            self.feature_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
        
        # Normal convolution to finalize task-dependent features
        self.final_task_conv = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1)
        
        # Task-specific bias adjustment
        # self.bias_fc = nn.Linear(task_vector_size, out_channels)
        
        self.task_scaling_weights = nn.Parameter(torch.tensor(0.01))
        # self.task_scaling_bias = nn.Parameter(torch.tensor(0.01))
        
        # Initialize layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.task_fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.final_task_conv.weight)
        nn.init.zeros_(self.final_task_conv.bias)
        # nn.init.xavier_uniform_(self.bias_fc.weight)
        # nn.init.xavier_uniform_(self.bias_fc.bias)
        # torch.nn.init.normal_(self.bias_fc.bias, mean=0.0, std=1.0)
    
    def forward(self, x, task_vector):
        # Get the base convolution weights and variance
        weight_var = torch.var(self.base_weights, dim=(1, 2, 3), keepdim=True)
        # print(weight_var.shape)
        
        task_vector = task_vector[0].unsqueeze(0)
        # Process task vector into feature map
        task_features = self.task_fc(task_vector)  # Shape: (batch_size, C*H*W)
        task_features = task_features.view(self.out_channels, self.feature_channels, self.kernel_size, self.kernel_size)
        
        # Apply transpose convolution to double spatial dimensions
        task_features = self.conv1(task_features)
        
        # Apply normal convolution
        task_features = self.final_task_conv(task_features)
        
        task_adjustment = torch.tanh(task_features)  # Apply tanh activation
        adjusted_weights = self.base_weights + self.task_scaling_weights * torch.sqrt(weight_var) * task_adjustment
        
        
        # Adjust bias
        # bias_var = torch.var(self.base_bias,keepdim=True) if self.base_bias is not None else None
        # # print(bias_var.shape)
        # bias_adjustment = self.bias_fc(task_vector)
        # bias_adjustment = torch.tanh(bias_adjustment)
        # adjusted_bias = self.base_bias + self.task_scaling_bias * torch.sqrt(bias_var) * bias_adjustment if self.base_bias is not None else None
        # adjusted_bias = adjusted_bias.squeeze(0)
        
        # Apply convolution with adjusted weights and bias
        out = torch.nn.functional.conv2d(x, adjusted_weights, bias=None, stride=self.stride, padding=self.padding)

        return out


        

