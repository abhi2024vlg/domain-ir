import random
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import DenoisingDataset, DerainingDataset, DeblurringDataset, LowLightDataset, UnifiedDataset
from architecture import RestormerWithNAFBlocks, D_NET
import numpy as np
from utils import GANLoss, SimpleMarginContrastiveLoss, generate_canvas, get_input, VGGPerceptualLoss
from lm import LanguageModel, LMHead
import json
import torch.backends.cudnn as cudnn
import os

# Your model setup

checkpoint_dir = "checkpoints"

# Create checkpoint directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)


cudnn.benchmark = True

def save_gradients(model, filename="gradients.txt"):
    with open(filename, "w") as f:
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_values = param.grad.cpu().numpy()
                f.write(f"Parameter: {name}\n")
                f.write(f"Gradient: {gradient_values}\n\n")
            else:
                f.write(f"Parameter: {name} has no gradient.\n\n")

# Modified to only include the four tasks we want
TASKS = {
    0: "restore_noise",      # Denoising
    1: "restore_rain",       # Deraining
    2: "restore_blur",       # Deblurring
    3: "restore_lowlight",   # Low-light enhancement
}

# Load only the datasets we need
denoise_ds     = DenoisingDataset("/home/cvpr_ug_2/DomainIR/noised_dataset")
derain_ds      = DerainingDataset("/home/cvpr_ug_2/DomainIR/rainy_image_dataset_train")
deblur_ds      = DeblurringDataset("/home/cvpr_ug_2/DomainIR/data/deblurring_dataset")
lowlight_ds    = LowLightDataset("/home/cvpr_ug_2/DomainIR/lol_dataset_train")

# Initialize the unified dataset with only our four datasets
dataset = UnifiedDataset(denoise_ds, derain_ds, deblur_ds, lowlight_ds)

RESTORATION_TASK_IDS = list(TASKS.keys())
     
num_con = len(TASKS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RestormerWithNAFBlocks(emb_dim=32, num_con=num_con).to(device)

discriminators = {task: D_NET(input_nc=3, ndf=32, block_num=3, norm_type='instance').to(device) for task in TASKS}
d_optimizers = {task: torch.optim.Adam(discriminators[task].parameters(), lr=1e-4) for task in TASKS}

LMODEL = 'TaylorAI/bge-micro-v2'
language_model = LanguageModel(model=LMODEL)
lm_head = LMHead(embedding_dim=384, hidden_dim=32, num_classes=num_con)
lm_head = lm_head.to(device)

PROMPTS_DATA = {}
with open("prompts.json", 'r', encoding='utf-8') as file:
    PROMPTS_DATA = json.load(file)

# Set learning rates
default_lr = 1e-4
task_conv_lr = 1e-5

# Collect parameters for different learning rates
task_conv_params = []
other_params = []

for name, param in model.named_parameters():
    if "task_conv" in name:
        task_conv_params.append(param)
    else:
        other_params.append(param)

# Define optimizer with different learning rates
optimizer = optim.Adam([
    {'params': other_params, 'lr': default_lr},
    {'params': task_conv_params, 'lr': task_conv_lr}
])

optimizer_lm = optim.Adam(lm_head.parameters(), lr=1e-3)
criterion = SimpleMarginContrastiveLoss(margin=0.05).to(device)
criterionGAN = GANLoss(device).to(device)

multiplier = 6
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
scheduler_lm = optim.lr_scheduler.StepLR(optimizer_lm, step_size=5, gamma=0.5)
criterion_class = nn.CrossEntropyLoss()
criterionVGG = VGGPerceptualLoss().to(device)

batch_size = 2
num_epochs = 400
num_epochs_lm = 7

# Training the language model head
for epoch in range(num_epochs_lm):
    for i in range(0, 2000):
        optimizer_lm.zero_grad()
        chosen_task_id = random.randint(0, num_con-1)
        chosen_task = TASKS[chosen_task_id]
        degradation_type = chosen_task.split("_")[1]

        c = np.zeros((batch_size, num_con))
        c[:, chosen_task_id] = 1

        prompt = random.choice(PROMPTS_DATA[degradation_type])
        lm_embd = language_model(prompt)
        lm_embd = lm_embd.to(device)
        text_embd, deg_pred = lm_head(lm_embd)

        #  Now shape [batch_size, num_classes]
        deg_pred = deg_pred.repeat(batch_size, 1)  
        
        c = torch.from_numpy(c).to(device, dtype=torch.float)
        c.requires_grad = False
        loss_class = criterion_class(deg_pred, c)

        loss_class.backward()
        optimizer_lm.step()

        print(f"Epoch {epoch}, Step {i // batch_size} task = {chosen_task}, Class_pred loss = {loss_class}")


optimizer_lm = optim.Adam(lm_head.parameters(), lr=1e-6)
scheduler_lm = optim.lr_scheduler.StepLR(optimizer_lm, step_size=5, gamma=0.5)

# Main training loop
for epoch in range(num_epochs):
    if epoch % 5 == 0:
        # Modified to only use the image types we care about
        img_HR, img_Noise, img_Rain, img_Blur, img_Lowlight = dataset.get_random_image_set(batch_size=batch_size)
        img_set = {
            "clean": img_HR,
            "noise": img_Noise,
            "rain": img_Rain,
            "blur": img_Blur,
            "lowlight": img_Lowlight
        }
        generate_canvas(model=model, epoch=epoch, img_set=img_set, num_con=num_con, TASKS=TASKS, device=device, lang_model=language_model, lmh=lm_head, pmd=PROMPTS_DATA)
    
    
    if (epoch + 1) % 100 == 0:
        # Save model weights
        checkpoint_path1 = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path1)
        
        # Save lm_head weights
        checkpoint_path2 = os.path.join(checkpoint_dir, f"lm_head_epoch_{epoch+1}.pth")
        torch.save(lm_head.state_dict(), checkpoint_path2)
        
        print(f"Both models' weights saved at epoch {epoch+1}")

    for i in range(0, 2000, batch_size):
        optimizer.zero_grad()
        optimizer_lm.zero_grad()
        for name, param in model.named_parameters():
            if name.startswith("task_conv"):
                param.requires_grad = True
       
        chosen_task_id = random.randint(0, num_con-1)
        chosen_task = TASKS[chosen_task_id]

        c = np.zeros((1, num_con))
        c[:, chosen_task_id] = 1

        # Get images for the four degradation types
        img_HR, img_Noise, img_Rain, img_Blur, img_Lowlight = dataset.get_random_image_set(batch_size=batch_size)
        img_set = {
            "clean": img_HR,
            "noise": img_Noise,
            "rain": img_Rain,
            "blur": img_Blur,
            "lowlight": img_Lowlight
        }
        
        # Restoration task
        degradation_type = chosen_task.split("_")[1]
        prompt = random.choice(PROMPTS_DATA[degradation_type])
        lm_embd = language_model(prompt)
        lm_embd = lm_embd.to(device)
        text_embd, deg_pred = lm_head(lm_embd)
        input_images = get_input(img_set=img_set, degradation_type=degradation_type)
        
        gt_images = img_set["clean"]  # ground-truth cleans
        pos_diff = gt_images - input_images

        # Forward pass for the chosen task
        inp = torch.from_numpy(input_images).to(device, dtype=torch.float).permute(0, 3, 1, 2)
        c = torch.from_numpy(c).to(device, dtype=torch.float)
        c.requires_grad = False
        loss_class = criterion_class(deg_pred, c)

        deg_pred_stacked = deg_pred.repeat(batch_size, 1)
        
        out_correct = model(inp, text_embd, deg_pred_stacked)
        anchor_diff = out_correct - inp

        gt = torch.from_numpy(gt_images).to(device, dtype=torch.float).permute(0, 3, 1, 2)
        pred_fake = discriminators[chosen_task_id](out_correct)
        
        gen_loss = criterionGAN(pred_fake, True)

        negative_task_ids = [t for t in RESTORATION_TASK_IDS if t != chosen_task_id]
        negative_diffs = []
        
        for t_id in negative_task_ids:
            t_name = TASKS[t_id]
            d = np.zeros((batch_size, num_con))
            d[:, t_id] = 1
            d = torch.from_numpy(d).to(device, dtype=torch.float)
            deg_type = t_name.split("_")[1]
            prompt_neg = random.choice(PROMPTS_DATA[deg_type])
            lm_embd_neg = language_model(prompt_neg)
            lm_embd_neg = lm_embd_neg.to(device)
            text_embd_neg, deg_pred_neg = lm_head(lm_embd_neg)
            deg_pred_neg = deg_pred_neg.repeat(batch_size, 1)  
            loss_class += criterion_class(deg_pred_neg, d)

            neg_inputs = get_input(img_set=img_set, degradation_type=deg_type)
            neg_input = torch.from_numpy(neg_inputs).to(device, dtype=torch.float).permute(0, 3, 1, 2)
            
            
            neg_out = model(neg_input, text_embd_neg, deg_pred_neg)

            neg_diff = neg_out - neg_input
            negative_diffs.append(neg_diff)

        pos_diff = torch.from_numpy(pos_diff).to(device, dtype=torch.float).permute(0, 3, 1, 2)

        # Compute loss
        loss = criterion(anchor_diff, pos_diff, negative_diffs)
        loss_perceptual, _, _ = criterionVGG(out_correct, gt)
        loss = multiplier*loss + 10*torch.nn.functional.l1_loss(out_correct, gt).to(device) + 4*gen_loss + 0.3*loss_class + loss_perceptual
        
        loss.backward()
        optimizer.step()
        optimizer_lm.step()

        if i % 240 == 0:
            save_gradients(lm_head)

        d_optimizer = d_optimizers[chosen_task_id]
        d_optimizer.zero_grad()

        pred_real = discriminators[chosen_task_id](gt)
        real_loss = criterionGAN(pred_real, True)

        pred_fake = discriminators[chosen_task_id](out_correct.detach())
        fake_loss = criterionGAN(pred_fake, False)

        d_loss = 2*(real_loss + fake_loss) 
        d_loss.backward()
        d_optimizer.step()

        print(f"Epoch {epoch}, Step {i // batch_size}: Loss = {loss.item():.4f}, task = {chosen_task}, GAN_Loss = {gen_loss}, Class_pred loss = {loss_class}")
        if i % 20 == 0:
            for name, param in model.named_parameters():
                if "task_scaling_weights" in name:  
                    print(f"{name}: {param.item()}")

    scheduler.step()
    scheduler_lm.step()

torch.save(lm_head .state_dict(), 'final_scaled_lm_head_state_dict.pth')
torch.save(model.state_dict(), 'final_scaled_model_state_dict.pth')
