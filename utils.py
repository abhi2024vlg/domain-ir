# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import random
from scipy.ndimage import gaussian_filter
import cv2
import torchvision

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


'''
ref. 
@article{chu2021tlsc,
  title={Revisiting Global Statistics Aggregation for Improving Image Restoration},
  author={Chu, Xiaojie and Chen, Liangyu and and Chen, Chengpeng and Lu, Xin},
  journal={arXiv preprint arXiv:2112.04491},
  year={2021}
}
'''
class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


from torch.autograd import Variable
from torchvision import models


#########  Loss  #####################
class GANLoss(nn.Module):
    def __init__(self,device, mse_loss=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = device
        if mse_loss:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor.to(self.device)

    def __call__(self, inputs, target_is_real):
        loss = 0.0
        if isinstance(inputs,list):
            for input in inputs:
                target_tensor = self.get_target_tensor(input, target_is_real)
                loss += self.loss(input, target_tensor)
        else:
            target_tensor = self.get_target_tensor(inputs, target_is_real)
            loss = self.loss(inputs, target_tensor)
        return loss
    
class SimpleMarginContrastiveLoss(nn.Module):
    def __init__(self, margin=0.05):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negatives):
    
        # Compute distances between anchor and positive
        dist_ap = torch.abs(anchor - positive).mean(dim=(1, 2, 3))  # Shape: (B,)

        # Concatenate negatives from the list
        negatives_tensor = torch.stack(negatives, dim=1)  # Shape: (B, N, C, H, W)
        B, N, C, H, W = negatives_tensor.shape

        # Reshape negatives for vectorized distance computation
        negatives_flat = negatives_tensor.reshape(B * N, C, H, W)  # Shape: (B*N, C, H, W)
        anchor_expanded = anchor.unsqueeze(1).repeat(1, N, 1, 1, 1).view(B * N, C, H, W)  # Shape: (B*N, C, H, W)

        # Compute distances between anchor and negatives
        dist_an = torch.abs(anchor_expanded - negatives_flat).mean(dim=(1, 2, 3))  # Shape: (B*N,)
        dist_an = dist_an.view(B, N)  # Reshape back to (B, N)

        # Aggregate negative distances (e.g., average)
        loss_neg = dist_an.sum(dim=1)  # Shape: (B,)

        # Compute contrastive loss
        loss = dist_ap / (self.margin + loss_neg)  # Add small epsilon to prevent division by zero

        return loss.mean()
    
def generate_img(img1, img2, A):
    img1 = (img1) * (1 - (img2 * 0.5 + 0.5)) + (A * (img2 * 0.5 + 0.5))
    return img1

def generate_haze(img1, img2):
    A = 0.8 + 0.2 * random.random()
    img1 = (img1) * (img2 * 0.5 + 0.5) + A * (1 - (img2 * 0.5 + 0.5))
    return img1

def generate_rain(img1, img2):
    A = 0.8 + 0.2 * random.random()
    img1 = (img1) * (1 - (img2 * 0.5 + 0.5)) + (A * (img2 * 0.5 + 0.5))
    return img1

def generate_snow(img1, img2):
    A = 0.8 + 0.2 * random.random()
    img1 = (img1) * (1 - (img2 * 0.5 + 0.5)) + (A * (img2 * 0.5 + 0.5))
    return img1

def add_noise(img):
    """
    Adds random Gaussian noise to the image.
    Increase the noise range for stronger noise.
    """
    # Increase the upper bound if you want more noise, e.g. 0.1 to 0.2
    noise_std = 0.1 + 0.1 * random.random()
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)

    img_noisy = img + noise
    # Clip to maintain [-1, 1]
    img_noisy = np.clip(img_noisy, -1.0, 1.0)
    return img_noisy

def add_blur(img):
    """
    Blurs only the spatial dimensions (H, W). The '0' in the channel position
    means we do not blur across the color channels, preserving color integrity.
    """
    sigma_spatial = 0.5 + 1.5 * random.random()
    # shape (H, W, C) -> (sigma_spatial, sigma_spatial, 0)
    img_blurred = gaussian_filter(img, sigma=(sigma_spatial, sigma_spatial, 0))
    return img_blurred

def get_raindrop(img, RD_1, RD_2):
    """
    Degrade an image with simulated raindrops.

    Args:
        img (np.array): Input image to degrade.
        RD_1 (np.array): Alpha mask for raindrop blending.
        RD_2 (np.array): Texture map for raindrops.

    Returns:
        np.array: Image degraded by raindrops.
    """
    RD_1 = (RD_1 * 0.5 + 0.5)
    RD_2 = (RD_2 * 0.5 + 0.5)

    position_matrix, alpha = get_position_matrix(RD_2, RD_1)
    img = np.transpose(img, (2, 1, 0))

    degraded_img = composition_img(img, alpha, position_matrix, rate=0.8 + 0.18 * random.random())
    return degraded_img

def get_position_matrix(texture, alpha):
    
    alpha = cv2.blur(alpha, (5, 5))
    position_matrix = np.mgrid[0:128, 0:128].astype(float)
    position_matrix[0, :, :] += texture[:, :, 2] * texture[:, :, 0]
    position_matrix[1, :, :] += texture[:, :, 1] * texture[:, :, 0]
    position_matrix = position_matrix * (alpha[:, :, 0] > 0.3)

    return position_matrix, alpha

def composition_img(img, alpha, position_matrix, rate, length=2):

    h, w = img.shape[:2]
    dis_img = (img * 0.5 + 0.5).copy()

    for x in range(h):
        for y in range(w):
            u, v = int(position_matrix[0, x, y] / length), int(position_matrix[1, x, y] / length)
            if u != 0 and v != 0:
                if (u < h) and (v < w):
                    dis_img[x, y, :] = dis_img[u, v, :]
                elif u < h:
                    dis_img[x, y, :] = dis_img[u, np.random.randint(0, w - 1), :]
                elif v < w:
                    dis_img[x, y, :] = dis_img[np.random.randint(0, h - 1), v, :]

    dis_img = cv2.blur(dis_img, (3, 3)) * rate
    alpha = np.transpose(alpha, (2, 1, 0))

    img = alpha * dis_img + (1 - alpha) * img
    return np.transpose(img, (2, 1, 0))

def create_random_holes(img, hole_count=1, min_size=10, max_size=30):
    
    img_with_holes = img.copy()
    h, w, _ = img.shape
    
    for _ in range(hole_count):
        hole_w = random.randint(min_size, max_size)
        hole_h = random.randint(min_size, max_size)
        x = random.randint(0, w - hole_w)
        y = random.randint(0, h - hole_h)
        
        img_with_holes[y:y+hole_h, x:x+hole_w, :] = -1  # Blacked-out region
    
    return img_with_holes
    
DEGRADATION_FUNCS = {
    "haze": generate_haze,
    "rain": generate_rain,
    "snow": generate_snow,
    "raindrop": get_raindrop,
    "blur": add_blur,
    "noise": add_noise,
}

def degrade_image(img1,degradation_type,img2 = None, img3 = None):
    if degradation_type == "blur" or degradation_type == "noise":
        return DEGRADATION_FUNCS[degradation_type](img1)
    elif degradation_type == "raindrop":
        return DEGRADATION_FUNCS[degradation_type](img1,img2,img3)
    return DEGRADATION_FUNCS[degradation_type](img1,img2)

# def get_input(img_set,degradation_type,batch_size):
#     if degradation_type == "inpaint":
#         input_images = np.stack([img_set["inpaint"][j] for j in range(batch_size)])
#     elif degradation_type == "blur" or degradation_type == "noise":
#         input_images = np.stack([degrade_image(img_set["clean"][j], degradation_type) for j in range(batch_size)])
#     elif degradation_type == "raindrop":
#         input_images = np.stack([degrade_image(img_set["clean"][j], degradation_type, img_set["rd1"][j], img_set["rd2"][j]) for j in range(batch_size)])
#     else:
#         input_images = np.stack([degrade_image(img_set["clean"][j], degradation_type, img_set[degradation_type][j]) for j in range(batch_size)])

#     return input_images
    
def get_input(img_set, degradation_type):
    """Return the appropriate degraded images based on degradation type"""
    if degradation_type == "noise":
        return img_set["noise"]
    elif degradation_type == "rain":
        return img_set["rain"]  
    elif degradation_type == "blur":
        return img_set["blur"]
    elif degradation_type == "lowlight":
        return img_set["lowlight"]
    else:
        # Default fallback
        return img_set["clean"]
    
def generate_canvas(model, epoch, img_set,num_con, TASKS, device,lang_model, lmh,pmd):
    num_tasks = num_con
    img_height, img_width = 128, 128
    canvas_height, canvas_width = img_height * num_tasks, img_width * 3
    canvas = Image.new('RGB', (canvas_width, canvas_height))

    for j in range(num_tasks):
        task_id = j
        task_name = TASKS[task_id]
        degradation_type = task_name.split("_")[1]
        
        # Take just first image from each batch for visualization
        img_set_single = {
            "clean": img_set["clean"][0:1],
            "noise": img_set["noise"][0:1],
            "rain": img_set["rain"][0:1],
            "blur": img_set["blur"][0:1], 
            "lowlight": img_set["lowlight"][0:1]
        }

        condition = np.zeros((1, num_con))
        condition[:, task_id] = 1

        prompt = random.choice(pmd[degradation_type])
        lm_embd = lang_model(prompt)
        lm_embd = lm_embd.to(device)
        text_embd, deg_pred = lmh(lm_embd)

        # Get input uses batch_size=1 here
        input_images = get_input(img_set=img_set_single, degradation_type=degradation_type)
        inp = torch.from_numpy(input_images).to(device, dtype=torch.float).permute(0, 3, 1, 2)
        c = torch.from_numpy(condition).to(device, dtype=torch.float)
        
        # Model processes batch_size=1
        output = model(inp, text_embd, deg_pred).squeeze(0).cpu().detach().numpy()
        
        output = ((output + 1) * 127.5).astype(np.uint8).transpose(1, 2, 0)
        output_image = Image.fromarray(output, mode='RGB')
        input_image = Image.fromarray(((input_images[0] + 1) * 127.5).astype(np.uint8), mode='RGB')
        gt_image = Image.fromarray(((img_set_single["clean"][0] + 1) * 127.5).astype(np.uint8), mode='RGB')
        
        canvas.paste(input_image, (0, img_height * j))
        canvas.paste(gt_image, (img_width, img_height * j))
        canvas.paste(output_image, (img_width * 2, img_height * j))
        
    canvas.save(f"output/comparison_{epoch}.png")
    torch.save(model.state_dict(), f"model_weights/model_state_dict_{epoch}.pth")

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, output, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        # if input.shape[1] != 3:
        #     # input = input.repeat(1, 3, 1, 1)
        #     target = target.repeat(1, 3, 1, 1)
        #     output = output.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        output = (output-self.mean) / self.std
        if self.resize:
            # input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            output = self.transform(output, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        loss1 = 0.0
        loss2 = 0.0

        loss_num = 0.0
        loss_den = 0.0
        loss_cont = 0.0

        # x = input
        y = target
        z = output
        for i, block in enumerate(self.blocks):
            # x = block(x)
            y = block(y)
            z = block(z)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(y, z)

                # loss_num = torch.nn.functional.l1_loss(y, z)
                # loss_den = torch.nn.functional.l1_loss(x, z)
                # loss_cont += loss_num / (loss_den)

                # z1 = z.mean([0,2,3])
                # y1 = y.mean([0,2,3])


                # z11 = z.std([0,2,3])
                # y11 = y.std([0,2,3])

                # loss1 += torch.nn.functional.l1_loss(z1, y1)
                # loss2 += torch.nn.functional.l1_loss(z11, y11)

            # if i in style_layers:
            #     act_z = z.reshape(z.shape[0], z.shape[1], -1)
            #     act_y = y.reshape(y.shape[0], y.shape[1], -1)
            #     gram_z = act_z @ act_z.permute(0, 2, 1)
            #     gram_y = act_y @ act_y.permute(0, 2, 1)
            #     loss += torch.nn.functional.l1_loss(gram_z, gram_y)
                
        return loss, loss1+loss2, loss_cont
