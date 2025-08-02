import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import os

def gen_images(diffusion_model, model, num_classes=10, images_per_class=2, image_shape=(3, 32, 32), device='cuda', save_dir='outputs'):
    """
    可视化并保存不同类别的扩散模型生成图像，这段代码是AI写的

    Args:
        diffusion_model: 已初始化好的 GaussianDiffusion 类实例
        model: 已加载权重的 denoise 网络 (如 U-Net)
        num_classes: 标签类别数，例如 CIFAR10 是 10
        images_per_class: 每个类别生成多少张图像
        image_shape: 单张图像的形状，例如 (3, 32, 32)
        device: 'cuda' 或 'cpu'
        save_dir: 图像保存目录
    """
    model.eval()
    model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    for label in range(num_classes):
        class_label_tensor = torch.full((images_per_class,), label, dtype=torch.long, device=device)
        full_image_shape = (images_per_class,) + image_shape  # (B, C, H, W)

        with torch.no_grad():
            images = diffusion_model.ddim_sample_loop(full_image_shape, device=device, label=class_label_tensor)

        images = images.clamp(0.0, 1.0).cpu()
        grid = make_grid(images, nrow=2)  # 2 行 4 列 或 1 行 8 列的网格

        # 显示
        plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.title(f"Class {label}")
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.savefig(f'diff.png', dpi=300, bbox_inches='tight')

        # 保存
        save_path = os.path.join(save_dir, f"class_{label}.png")
        save_image(grid, save_path)
        print(f"Saved class {label} images to {save_path}")
