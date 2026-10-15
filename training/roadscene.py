import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models.segmentation as segmentation
from torchvision import transforms
from transformers import CLIPTokenizer

# ==========================================
# 引入项目模块 (请确保路径正确)
# ==========================================
from model.Text_IF_model import Text_IF
from training.roadscene_dataset import RoadSceneDataset
from training.losses import PatentLoss
from training.semantic_loss import SemanticLoss
from training.utils import create_lr_scheduler, setup_seed


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 扫描图片对
    exts = ['.jpg', '.png', '.bmp']
    files = sorted([f for f in os.listdir(args.ir_path) if os.path.splitext(f)[-1].lower() in exts])
    valid_list = [{'file_name': f} for f in files if os.path.exists(os.path.join(args.vis_path, f))]

    # 划分 9:1 训练验证
    split = int(len(valid_list) * 0.9)
    train_set, val_set = valid_list[:split], valid_list[split:]

    # 2. 初始化 Dataset
    train_ds = RoadSceneDataset(train_set, args.ir_path, args.vis_path,
                                mask_folder=args.mask_path,
                                img_size=256, mode='train', prompt_json_path=args.prompt_path)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 3. 加载 Fusion 模型
    model = Text_IF(inp_A_channels=1, inp_B_channels=3).to(device)
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        print(f"�� 加载预训练权重: {args.pretrained_weights}")
        ckpt = torch.load(args.pretrained_weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get('model', ckpt), strict=False)

    # 4. 初始化辅助 Judge 模型 (MSRS 9分类)
    # 启用 aux_loss=True 以匹配 best_judge.pth 结构
    judge = segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
    num_classes = 9
    judge.classifier[4] = nn.Conv2d(256, num_classes, 1)
    judge.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)

    if os.path.exists(args.judge_weights):
        print(f"✅ 加载 Judge 权重以支持语义损失: {args.judge_weights}")
        judge_ckpt = torch.load(args.judge_weights, map_location=device, weights_only=False)
        judge_state = judge_ckpt.get('model', judge_ckpt)
        # 自动移除 module. 前缀
        judge_state = {k.replace('module.', ''): v for k, v in judge_state.items()}
        judge.load_state_dict(judge_state, strict=False)
    else:
        print("⚠️ 警告: 未找到 Judge 权重！")

    judge.to(device).eval()
    for param in judge.parameters():
        param.requires_grad = False

    # 5. 损失函数与归一化
    loss_fn = PatentLoss(device)
    semantic_loss_fn = SemanticLoss(ignore_index=255).to(device)
    # Judge 专用的归一化 (针对 3 通道)
    judge_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    # 6. 日志与保存路径
    save_dir = f"{args.output_root}/RoadScene_FullFixed_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(f"{save_dir}/weights", exist_ok=True)
    writer = SummaryWriter(f"{save_dir}/log")

    print(f"�� 开始满血训练 | 过曝约束: 开启 | 内部注意力: 开启 | Sem权重: {args.lambda_semantic}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            ir, vis = data['ir'].to(device), data['vis'].to(device)
            ids, att = data['input_ids'].to(device), data['attention_mask'].to(device)
            mask = data.get('mask').to(device)  # 伪标签

            optimizer.zero_grad()

            # --- Forward ---
            output = model(ir, vis, ids, att)
            if isinstance(output, tuple):
                fused, pred_mask_internal = output[0], output[1]
            else:
                fused, pred_mask_internal = output, None

            # --- 核心约束 1: 防止数值爆炸导致洗白 ---
            fused = torch.clamp(fused, -1.02, 1.02)
            if len(fused.shape) == 3: fused = fused.unsqueeze(1)

            # --- Loss A: 物理融合损失 ---
            loss_fusion = loss_fn(fused, ir, vis, ids, att)

            # --- Loss B: 语义裁判损失 (修正通道报错) ---
            fused_01 = (fused + 1) / 2.0
            # [核心修复]：判断通道，确保进入 judge_norm 的是 3 通道
            if fused_01.shape[1] == 1:
                fused_rgb = fused_01.repeat(1, 3, 1, 1)
            else:
                fused_rgb = fused_01

            judge_input = judge_norm(fused_rgb)
            with torch.no_grad():
                seg_output = judge(judge_input)['out']
            loss_semantic = semantic_loss_fn(seg_output, mask.long())

            # --- Loss C: 内部注意力监督 (Aux Loss) ---
            loss_aux = torch.tensor(0.0).to(device)
            if pred_mask_internal is not None:
                # 9分类转二值 (0是背景, >0是物体)
                gt_binary = (mask > 0).float().unsqueeze(1)
                if gt_binary.shape[-2:] != pred_mask_internal.shape[-2:]:
                    gt_binary = F.interpolate(gt_binary, size=pred_mask_internal.shape[-2:], mode='nearest')
                loss_aux = F.binary_cross_entropy(pred_mask_internal, gt_binary)

            # --- 核心约束 2: 亮度均值惩罚 (防止画面过曝) ---
            loss_int_penalty = torch.mean(torch.pow(fused, 2))

            # --- 总损失结合 ---
            total_loss = loss_fusion + args.lambda_semantic * loss_semantic + 0.2 * loss_aux + 0.05 * loss_int_penalty

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] Total: {total_loss.item():.4f} | "
                      f"Fus: {loss_fusion.item():.4f} | Sem: {loss_semantic.item():.4f} | "
                      f"Aux: {loss_aux.item():.4f} | IntP: {loss_int_penalty.item():.4f}")

        scheduler.step()
        writer.add_scalar("Train/Total_Loss", running_loss / len(train_loader), epoch)

        # 覆写保存逻辑
        save_path = f"{save_dir}/weights/latest_model.pth"
        torch.save({'model': model.state_dict()}, save_path)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"✅ 模型已覆写保存至 {save_path}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    # RoadScene 建议此权重保持在 0.02 左右，过高容易导致颜色失真
    parser.add_argument('--lambda_semantic', type=float, default=0.02)

    # 路径配置 (请确保这些路径在你的服务器上正确)
    parser.add_argument('--ir_path', type=str, default="./data/RoadScene_Train/ir")
    parser.add_argument('--vis_path', type=str, default="./data/RoadScene_Train/vi")
    parser.add_argument('--mask_path', type=str, default="./data/RoadScene_Train/masks/pseudo_labels")
    parser.add_argument('--prompt_path', type=str,
                        default="./data/RoadScene_Train/road_train_prompt.json")
    parser.add_argument('--judge_weights', type=str, default="./weights/best_judge.pth")
    parser.add_argument('--pretrained_weights', type=str, default="")

    parser.add_argument("--output-root", default="./experiments", help="Parent directory for the original timestamped run folders.")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())