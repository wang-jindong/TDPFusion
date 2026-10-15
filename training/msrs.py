import os
import argparse
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTokenizer
import warnings
import torchvision
import torchvision.transforms as T
import torchvision.models.segmentation as segmentation
import re

# === Ignore unrelated warnings ===
torchvision.disable_beta_transforms_warning()
warnings.filterwarnings("ignore")

# ==========================================
# Import Project Modules
# ==========================================
# Ensure your folder structure contains these files
from model.Text_IF_model import Text_IF
from training.patent_dataset import FusionDataset
from training.losses import PatentLoss
# Ensure scripts/semantic_loss.py exists
from training.semantic_loss import SemanticLoss
from training.utils import create_lr_scheduler, setup_seed


# ==========================================
# Helper Functions
# ==========================================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def scan_images(ir_folder, vis_folder):
    """
    Scans the infrared folder and checks if corresponding visible images exist.
    """
    exts = ['.jpg', '.png', '.bmp', '.tif', '.jpeg']
    if not os.path.exists(ir_folder):
        raise FileNotFoundError(f"IR folder not found: {ir_folder}")

    files = os.listdir(ir_folder)
    valid_files = []
    # print(f"DEBUG: Scanning folder {ir_folder} ...")
    for f in files:
        if os.path.splitext(f)[-1].lower() in exts:
            if os.path.exists(os.path.join(vis_folder, f)):
                valid_files.append({'file_name': f})
    return valid_files


# ==========================================
# [Chapter 4 Core] Load Judge Model
# ==========================================
def load_judge_model(weights_path, device):
    """
    Loads a pre-trained DeepLabV3 model to act as a fixed semantic judge.
    """
    print(f"[Judge] Loading judge model from {weights_path}...")

    # === Key: Must enable aux_loss=True to match architecture ===
    model = segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)

    # Modify classifier for 9 classes (MSRS dataset standard)
    model.classifier[4] = nn.Conv2d(256, 9, kernel_size=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, 9, kernel_size=(1, 1))

    # Load weights
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("✅ Judge weights loaded successfully!")
    else:
        print(f"❌ Error: Judge weights not found at {weights_path}!")
        print("Please train a segmentation model first or check the path.")
        exit()  # Exit if judge is missing

    # === Core: Freeze the Judge ===
    # We only train the fusion network; the judge must remain static!
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model.to(device)


# ==========================================
# Training Logic (Ours: Text Guided + Judge + Mask Aux)
# ==========================================
def train_one_epoch(model, optimizer, lr_scheduler, train_loader, device, epoch,
                    fusion_loss_fn, semantic_loss_fn, judge_model,
                    tokenizer, args, prompt_map_id):
    model.train()
    judge_model.eval()  # Judge is always in eval mode

    running_loss = 0.0
    optimizer.zero_grad()
    accumulation_steps = getattr(args, 'accumulation_steps', 1)

    # Default text prompt fallback
    default_text = "infrared and visible image fusion"

    # Normalization required by Judge (ImageNet Mean/Std)
    judge_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for batch_idx, data in enumerate(train_loader):
        # 1. Unpack Data
        if 'ir' in data:
            # Input range is [-1, 1]
            ir_img = data['ir'].to(device)
            vis_img = data['vis'].to(device)
            current_filenames = data.get('file_name', [])

            # Get Segmentation Mask (GT) if available
            if 'mask' in data:
                gt_mask = data['mask'].to(device)
            else:
                gt_mask = None
        else:
            raise KeyError("Dataset Keys Error: Dataset must provide 'ir' and 'vis'")

        # 2. Dynamic Text Embeddings (ID Matching)
        batch_prompts = []
        if len(current_filenames) > 0:
            for fname in current_filenames:
                text = None
                # Extract numeric ID from filename
                nums = re.findall(r'\d+', fname)
                if len(nums) > 0:
                    f_id = nums[-1]
                    text = prompt_map_id.get(f_id)

                if text is None:
                    text = default_text
                batch_prompts.append(text)
        else:
            batch_prompts = [default_text] * ir_img.size(0)

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 3. Forward Pass
        # Text_IF returns: (fused_img, pred_mask_internal)
        # pred_mask_internal is the attention map from SSGA module
        output = model(ir_img, vis_img, input_ids, attention_mask)

        if isinstance(output, tuple):
            fused_img = output[0]
            pred_mask_internal = output[1]
        else:
            fused_img = output
            pred_mask_internal = None

        # 4. Dimension Check & Fix
        if len(fused_img.shape) == 3:
            fused_img = fused_img.unsqueeze(1)
        if len(ir_img.shape) == 3:
            ir_img = ir_img.unsqueeze(1)
        if len(vis_img.shape) == 3:
            vis_img = vis_img.unsqueeze(1)

        # 5. Calculate Losses

        # A. Basic Fusion Loss
        loss_fusion = fusion_loss_fn(fused_img, ir_img, vis_img, input_ids, attention_mask)

        # B. [Chapter 4 Core] Semantic Judge Loss
        loss_semantic = torch.tensor(0.0).to(device)

        if gt_mask is not None and args.lambda_semantic > 0:
            # === Value Bridging ===
            # Convert [-1, 1] -> [0, 1] for the judge
            judge_input_01 = (fused_img + 1) / 2.0
            judge_input_01 = torch.clamp(judge_input_01, 0, 1)

            # Convert 1-channel to 3-channel
            if judge_input_01.shape[1] == 1:
                judge_input_01 = judge_input_01.repeat(1, 3, 1, 1)
            elif judge_input_01.shape[1] == 3:
                pass

                # Normalize and feed to Judge
            judge_input = judge_normalize(judge_input_01)

            with torch.no_grad():
                judge_output = judge_model(judge_input)
                seg_logits = judge_output['out']

            # Calculate penalty (Cross Entropy)
            loss_semantic = semantic_loss_fn(seg_logits, gt_mask)

        # C. [New] Internal Mask Auxiliary Loss
        # Supervises the attention map using GT (Foreground vs Background)
        loss_aux_mask = torch.tensor(0.0).to(device)

        if gt_mask is not None and pred_mask_internal is not None:
            # GT Mask Processing: MSRS 0 is background, >0 is object
            # Convert to binary map (Foreground vs Background)
            gt_binary = (gt_mask > 0).float().unsqueeze(1)  # [B, 1, H, W]

            # Resize GT to match pred_mask_internal size
            if gt_binary.shape[-2:] != pred_mask_internal.shape[-2:]:
                gt_binary_small = F.interpolate(gt_binary, size=pred_mask_internal.shape[-2:], mode='nearest')
            else:
                gt_binary_small = gt_binary

            # Calculate Binary Cross Entropy
            loss_aux_mask = F.binary_cross_entropy(pred_mask_internal, gt_binary_small)

        # 6. Total Loss
        # Recommendation: Weight aux_mask at 0.2 to guide attention without dominating
        loss = loss_fusion + args.lambda_semantic * loss_semantic + 0.2 * loss_aux_mask

        # Gradient Accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

        # Logging
        if batch_idx % 20 == 0:
            real_loss = loss.item() * accumulation_steps
            print(f"Epoch [{epoch}] Batch [{batch_idx}] "
                  f"Total: {real_loss:.4f} | Fusion: {loss_fusion.item():.4f} | "
                  f"Sem: {loss_semantic.item():.4f} | AuxMask: {loss_aux_mask.item():.4f}")

    if lr_scheduler is not None:
        lr_scheduler.step()

    return running_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, dataloader, device, fusion_loss_fn, tokenizer, args, prompt_map_id):
    model.eval()
    running_loss = 0.0
    default_text = "infrared and visible image fusion"

    for data in dataloader:
        ir_img = data['ir'].to(device)
        vis_img = data['vis'].to(device)
        current_filenames = data.get('file_name', [])

        # Dynamic Prompts
        batch_prompts = []
        if len(current_filenames) > 0:
            for fname in current_filenames:
                text = None
                nums = re.findall(r'\d+', fname)
                if len(nums) > 0:
                    f_id = nums[-1]
                    text = prompt_map_id.get(f_id)
                if text is None:
                    text = default_text
                batch_prompts.append(text)
        else:
            batch_prompts = [default_text] * ir_img.size(0)

        inputs = tokenizer(batch_prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        output = model(ir_img, vis_img, input_ids, attention_mask)
        if isinstance(output, tuple):
            fused_img = output[0]
        else:
            fused_img = output

        # Dimension check
        if len(fused_img.shape) == 3: fused_img = fused_img.unsqueeze(1)
        if len(ir_img.shape) == 3: ir_img = ir_img.unsqueeze(1)
        if len(vis_img.shape) == 3: vis_img = vis_img.unsqueeze(1)

        loss = fusion_loss_fn(fused_img, ir_img, vis_img, input_ids, attention_mask)
        running_loss += loss.item()

    return running_loss / len(dataloader)
def main(args):
    setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Log Path - Marked as Ours (With Judge + Aux)
    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = f"{args.output_root}/TextIF_Ours_Full_{file_name}"
    os.makedirs(f"{filefold_path}/weights", exist_ok=True)
    os.makedirs(f"{filefold_path}/log", exist_ok=True)
    tb_writer = SummaryWriter(log_dir=f"{filefold_path}/log")

    # ==========================================
    # Load Prompts (Prompt Map)
    # ==========================================
    print(f"Loading prompts from {args.prompt_path} ...")
    prompt_map_id = {}
    import json
    if os.path.exists(args.prompt_path):
        with open(args.prompt_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for item in json_data:
                fname = item.get('file_name')
                text = item.get('text', item.get('caption', item.get('description', "")))
                if fname:
                    nums = re.findall(r'\d+', fname)
                    if len(nums) > 0:
                        file_id = nums[-1]
                        prompt_map_id[file_id] = text
        print(f"Loaded {len(prompt_map_id)} prompts.")
    else:
        print(f"Warning: Prompt file {args.prompt_path} not found! Using default prompts.")
    # Data Prep
    full_list = scan_images(args.ir_path, args.vis_path)
    train_list = full_list[:int(len(full_list) * 0.9)]
    val_list = full_list[int(len(full_list) * 0.9):]
    print(f"Total training images: {len(train_list)}")
    # Initialize Dataset
    train_dataset = FusionDataset(
        train_list, args.ir_path, args.vis_path,
        mask_folder=args.mask_path,  # Must provide Mask path for Judge
        img_size=256, mode='train'
        # prompt_json_path logic handled manually in train_loop via prompt_map_id
    )
    val_dataset = FusionDataset(val_list, args.ir_path, args.vis_path, img_size=256, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # Initialize Model
    print("Initializing Fusion Model...")
    model = Text_IF(inp_A_channels=1, inp_B_channels=3).to(device)
    # Initialize Judge
    # Ensure this path points to your trained segmentation model
    judge_weights = args.judge_weights
    judge_model = load_judge_model(judge_weights, device)
    # Initialize Semantic Loss
    semantic_loss_fn = SemanticLoss(ignore_index=255).to(device)
    # Initialize Tokenizer
    try:
        tokenizer = CLIPTokenizer.from_pretrained(os.environ.get("TDPFUSION_CLIP_PATH", "./clip_weights"))
        print("[Info] Loaded local CLIP tokenizer.")
    except:
        print("[Info] Downloading CLIP tokenizer from HuggingFace...")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    loss_fn = PatentLoss(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, optimizer, lr_scheduler, train_loader, device, epoch,
            loss_fn, semantic_loss_fn, judge_model,
            tokenizer, args, prompt_map_id
        )
        tb_writer.add_scalar("Train/Total_Loss", train_loss, epoch)
        print(f"Epoch [{epoch}/{args.epochs}] Train Loss: {train_loss:.4f}")
        if epoch % args.val_every_epoch == 0:
            val_loss = evaluate(model, val_loader, device, loss_fn, tokenizer, args, prompt_map_id)
            tb_writer.add_scalar("Val/Loss", val_loss, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model': model.state_dict()}, f"{filefold_path}/weights/best_checkpoint.pth")
                print(f"Best model saved!")
def build_parser():
    parser = argparse.ArgumentParser()
    # Basic Params
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    # [Core] Loss Weights
    parser.add_argument('--lambda_semantic', type=float, default=0.1, help="Weight for external Judge semantic loss")
    # Paths
    parser.add_argument('--ir_path', type=str, default="./data/MSR_Train/ir")
    parser.add_argument('--vis_path', type=str, default="./data/MSR_Train/vi")
    # Mask Path (Critical for Judge)
    parser.add_argument('--mask_path', type=str, default="./data/MSR_Train/Label")
    parser.add_argument('--prompt_path', type=str,
                        default="./data/MSR_Train/MSR_c4_train_res.json")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--val_every_epoch', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--output-root", default="./experiments", help="Parent directory for the original timestamped run folders.")
    parser.add_argument("--judge_weights", default="./weights/best_judge.pth")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())