#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# 模仿原代码中的 PILtoTorch 逻辑，但适配 transformers processor
def get_scaled_resolution(orig_w, orig_h, resolution=1):
    """
    模仿 Inria 逻辑：如果宽度超过 1600，则缩放到 1600
    """
    if resolution == 1:
        if orig_w > 1600:
            global_down = orig_w / 1600
        else:
            global_down = 1.0
    else:
        global_down = orig_w / resolution

    scale = float(global_down)
    new_width = int(orig_w / scale)
    new_height = int(orig_h / scale)
    return (new_width, new_height)

def colormap_depth(depth_map, cmap='magma'):
    """
    将深度图转换为伪彩色（可视化）
    """
    import matplotlib.pyplot as plt
    # 归一化到 0-1
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    
    # 使用 matplotlib 应用颜色映射
    cm = plt.get_cmap(cmap)
    depth_color = (cm(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    return depth_color

def process(input_path, output_path, model_id, device):
    # 1. 加载 transformers 模型和处理器
    print(f"[ INFO ] Loading model from Hugging Face: {model_id}")
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    print(f"[ INFO ] Processing {len(image_files)} images...")

    for img_name in tqdm(image_files):
        img_full_path = os.path.join(input_path, img_name)
        
        # 加载原始图片
        raw_image = Image.open(img_full_path).convert("RGB")
        orig_w, orig_h = raw_image.size

        # 2. 计算 1.6K 缩放分辨率
        target_resolution = get_scaled_resolution(orig_w, orig_h)
        
        # 缩放图片 (模仿 loadCam 的 resize 行为)
        resized_image = raw_image.resize(target_resolution, Image.LANCZOS)

        # 3. 准备输入
        inputs = image_processor(images=resized_image, return_tensors="pt").to(device)

        # 4. 推理
        with torch.no_grad():
            outputs = model(**inputs)
            # 模型输出的是预测的深度（通常是相对深度）
            predicted_depth = outputs.predicted_depth

        # 5. 插值回目标分辨率 (即 1.6K 后的分辨率)
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=target_resolution[::-1], # 转换 (W,H) 为 (H,W)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        # 6. 保存原始深度和可视化结果
        raw_depth_file = os.path.join(output_path, os.path.splitext(img_name)[0] + "_depth.npy")
        np.save(raw_depth_file, depth_map)

        depth_viz = colormap_depth(depth_map)
        output_file = os.path.join(output_path, os.path.splitext(img_name)[0] + "_depth.png")
        Image.fromarray(depth_viz).save(output_file)

        print(f"[ INFO ] Saved raw depth to {raw_depth_file} and visualization to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str, required=True)
    parser.add_argument("--output_path", "-o", type=str, required=True)
    parser.add_argument("--model_id", "-m", type=str, default="depth-anything/Depth-Anything-V2-Large-hf")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模仿 safe_state 执行核心逻辑
    process(args.input_path, args.output_path, args.model_id, device)
    
    print(f"[ INFO ] All images processed. Results in: {args.output_path}")