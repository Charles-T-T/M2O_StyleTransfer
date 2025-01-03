import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torchvision import models
from torch.autograd import Variable
import numpy as np
import torch.distributed as dist
import os
from torch.cuda.amp import GradScaler, autocast

# Transformer class to modify layers as needed
class transformer(nn.Module):
    def __init__(self, subnet):
        super(transformer, self).__init__()
        self.net = subnet
        for name, layer in self.net.named_children():
            if isinstance(layer, nn.MaxPool2d):
                self.net[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)

    def get_gram_matrix(self, style):
        """
        计算风格特征的Gram矩阵
        style: 特征图列表，每个元素shape为[1, C, H, W]
        """
        gram_matrix = []
        for feature in style:
            b, c, h, w = feature.size()  # 获取4个维度
            feature = feature.view(b * c, h * w)  # 重塑为2D矩阵
            gram = torch.mm(feature, feature.t())  # 计算Gram矩阵
            gram = gram.div(c * h * w)  # 归一化
            gram_matrix.append(gram)
        return gram_matrix

    def forward(self, x, content_list=None, style_list=None):
        content = []
        style = []
        
        for name, layer in self.net.named_children():
            x = layer(x)
            if content_list and int(name) in content_list:
                content.append(x)
            if style_list and int(name) in style_list:
                style.append(x)

        # Compute Gram matrix for style loss
        style_matrix = self.get_gram_matrix(style)
        return content, style_matrix


# Training function for style transfer
class MaskGenerator:
    def __init__(self, num_regions):
        self.num_regions = num_regions
    
    def create_sky_ground_mask(self, image):
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        binary = cv2.adaptiveThreshold(
            l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 201, 2
        )
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 使用GrabCut进行精细分割
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        
        # 初始化矩形区域
        height, width = image.shape[:2]
        rect = (0, 0, width, height//2)
        
        # 运行GrabCut算法
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # 创建基础遮罩
        height, width = image.shape[:2]
        sky_mask_np = np.where((mask==2)|(mask==0), 0, 1).astype('float32')
        
        # 创建过渡区域
        transition_width = 30  # 过渡区域的宽度
        
        # 使用距离变换创建渐变
        dist = cv2.distanceTransform(sky_mask_np.astype(np.uint8), cv2.DIST_L2, 5)
        dist_inv = cv2.distanceTransform((1-sky_mask_np).astype(np.uint8), cv2.DIST_L2, 5)
        
        # 标准化距离
        dist = dist / (dist + dist_inv + 1e-6)
        
        # 应用sigmoid函数使过渡更平滑
        dist = 1 / (1 + np.exp(-(dist - 0.5) * 8))
        
        # 使用高斯模糊进一步平滑
        blur_radius = 15
        sky_mask_np = cv2.GaussianBlur(dist, (blur_radius, blur_radius), 0)
        
        # 创建遮罩
        masks = []
        
        # 创建天空遮罩
        sky_mask = torch.zeros((3, height, width))
        for c in range(3):
            sky_mask[c] = torch.from_numpy(sky_mask_np)
        masks.append(sky_mask)
        
        # 创建地面遮罩（使用平滑的补集）
        ground_mask = torch.zeros((3, height, width))
        ground_mask_np = 1 - sky_mask_np
        for c in range(3):
            ground_mask[c] = torch.from_numpy(ground_mask_np)
        masks.append(ground_mask)
        
        # 可视化分割结果
        if True:
            plt.figure(figsize=(20, 5))
            plt.subplot(141)
            plt.imshow(image_rgb)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(142)
            plt.imshow(sky_mask_np, cmap='gray')
            plt.title('Sky Mask with Smooth Transition')
            plt.axis('off')
            
            plt.subplot(143)
            plt.imshow(ground_mask_np, cmap='gray')
            plt.title('Ground Mask with Smooth Transition')
            plt.axis('off')
            
            # 显示过渡区域
            transition = np.abs(sky_mask_np - 0.5) < 0.2
            plt.subplot(144)
            plt.imshow(transition, cmap='hot')
            plt.title('Transition Area')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return masks
    
    def create_grid_mask(self, height, width):
        """创建网格状的遮掩"""
        rows = int(np.sqrt(self.num_regions))
        cols = self.num_regions // rows if self.num_regions % rows == 0 else rows
        masks = []
        h_step = height // rows
        w_step = width // cols
        
        for i in range(rows):
            for j in range(cols):
                # 创建3通道的遮罩
                mask = torch.zeros((3, height, width))
                mask[:, i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step] = 1
                masks.append(mask)
                
                if len(masks) == self.num_regions:  # 确保只创建需要的数量的遮罩
                    return masks
        return masks
    
    def create_random_mask(self, height, width):
        """创建随机形状的遮罩"""
        masks = []
        for _ in range(self.num_regions):
            # 创建3通道的遮罩
            mask = torch.rand(3, height, width) > 0.5
            masks.append(mask.float())
        return masks
    
    def create_smooth_transition(self, mask, blur_radius=15):
        """
        为遮罩创建平滑过渡
        """
        # 确保mask是numpy数组
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
        
        # 归一化到[0,1]范围
        blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min())
        
        return torch.from_numpy(blurred).float()
    
    def create_multi_object_mask(self, image, num_segments=3):
        """
        创建多个物体区域的遮罩，使用改进的分割方法保持物体完整性
        image: 原始图像 [H, W, C]
        num_segments: 期望分割的区域数量
        """
        # 转换为RGB格式
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 使用SLIC超像素分割
        from skimage.segmentation import slic, mark_boundaries
        from skimage.feature import canny
        from scipy import ndimage as ndi
        
        # 步骤1: 使用SLIC生成超像素
        segments = slic(image_rgb, n_segments=100, compactness=10, sigma=1)
        
        # 步骤2: 边缘检测
        edges = canny(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY))
        
        # 步骤3: 使用分水岭算法进行区域合并
        distance = ndi.distance_transform_edt(~edges)
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        
        # 找到局部最大值作为标记点
        coordinates = peak_local_max(distance, min_distance=20)
        local_max = np.zeros_like(distance, dtype=bool)
        local_max[tuple(coordinates.T)] = True
        markers = ndi.label(local_max)[0]
        
        # 应用分水岭算法
        labels = watershed(-distance, markers, mask=~edges)
        
        # 步骤4: 合并小区域
        from skimage.measure import regionprops
        props = regionprops(labels)
        
        # 计算区域大小的中位数
        areas = [prop.area for prop in props]
        median_area = np.median(areas)
        
        # 合并小区域到邻近的大区域
        new_labels = labels.copy()
        for prop in props:
            if prop.area < median_area * 0.3:  # 小于中位数面积30%的区域被视为小区域
                mask = labels == prop.label
                dilated = ndi.binary_dilation(mask)
                neighbors = np.unique(labels[dilated & ~mask])
                if len(neighbors) > 0 and 0 not in neighbors:  # 确保不会合并到背景
                    # 合并到最大的邻居区域
                    largest_neighbor = max(neighbors, key=lambda x: np.sum(labels == x))
                    new_labels[mask] = largest_neighbor
        
        # 步骤5: 使用K-means对区域进行聚类得到最终的num_segments个区域
        from sklearn.cluster import KMeans
        
        # 为每个区域计算颜色和位置特征
        features = []
        unique_labels = np.unique(new_labels)
        unique_labels = unique_labels[unique_labels != 0]  # 排除背景标签
        
        for label in unique_labels:
            mask = new_labels == label
            if np.sum(mask) > 0:  # 确保区域非空
                color_mean = np.mean(image_rgb[mask], axis=0)
                position = np.mean(np.where(mask), axis=1)
                # 增加颜色特征的权重
                features.append(np.concatenate([color_mean * 2, position * 0.5]))
        
        if len(features) < num_segments:
            print(f"Warning: Only found {len(features)} regions, but {num_segments} were requested.")
            num_segments = max(1, len(features))
        
        # 聚类得到最终区域
        kmeans = KMeans(n_clusters=num_segments, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # 创建最终的遮罩
        height, width = image_rgb.shape[:2]
        masks = []
        
        # 为每个最终区域创建遮罩
        for segment in range(num_segments):
            base_mask = np.zeros((height, width), dtype=np.float32)
            for orig_label, cluster in zip(unique_labels, cluster_labels):
                if cluster == segment:
                    base_mask[new_labels == orig_label] = 1
            
            # 使用形态学操作清理遮罩
            kernel = np.ones((5,5), np.uint8)
            base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel)
            base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel)
            
            # 使用距离变换创建平滑过渡
            dist = cv2.distanceTransform(base_mask.astype(np.uint8), cv2.DIST_L2, 5)
            dist_inv = cv2.distanceTransform((1-base_mask).astype(np.uint8), cv2.DIST_L2, 5)
            
            # 标准化距离并创建平滑过渡
            dist = dist / (dist + dist_inv + 1e-6)
            smooth_mask = 1 / (1 + np.exp(-(dist - 0.5) * 8))
            smooth_mask = cv2.GaussianBlur(smooth_mask, (15, 15), 0)
            
            # 创建三通道遮罩
            mask = torch.zeros((3, height, width))
            for c in range(3):
                mask[c] = torch.from_numpy(smooth_mask)
            masks.append(mask)
        
        # 可视化分割结果
        if True:
            plt.figure(figsize=(5 * (num_segments + 2), 5))
            
            # 显示原图
            plt.subplot(1, num_segments + 2, 1)
            plt.imshow(image_rgb)
            plt.title('Original Image')
            plt.axis('off')
            
            # 显示边缘检测结果
            plt.subplot(1, num_segments + 2, 2)
            plt.imshow(edges, cmap='gray')
            plt.title('Edges')
            plt.axis('off')
            
            # 显示每个区域的遮罩
            for i in range(num_segments):
                plt.subplot(1, num_segments + 2, i + 3)
                plt.imshow(masks[i][0].numpy(), cmap='gray')
                plt.title(f'Region {i+1} Mask')
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return masks

# 修改 train 函数以支持多风格和遮掩
def train(x, net, parameter_list, content, styles):
    """
    x: 输入图像 [1, 3, H, W]
    styles: 列表，含多个风格的 Gram 矩阵
    """
    epoches = parameter_list["epoches"]
    device = parameter_list["device"]
    content_list = parameter_list["content_list"]
    style_list = parameter_list["style_list"]

    # 创建遮罩
    mask_generator = MaskGenerator(num_regions=len(styles))
    content_img = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    content_img = (content_img * 255).astype(np.uint8)
    # 使用新的多物体分割方法
    masks = mask_generator.create_multi_object_mask(content_img, num_segments=len(styles))
    masks = [mask.to(device) for mask in masks]

    x.requires_grad = True
    x = x.to(device)
    net.to(device)

    optimizer = optim.Adam([x], lr=parameter_list["lr"])

    # FHT: 使用AdamW优化器
    # optimizer = optim.AdamW([x], lr=parameter_list["lr"])

    for epoch in range(epoches):
        optimizer.zero_grad()

        # 计算内容损失
        x_content, x_style = net(x, content_list, style_list)
        content_loss = torch.tensor(0.0, device=device)
        for i in range(len(x_content)):
            content_loss = content_loss + F.mse_loss(x_content[i], content[i])
        style_losses = [] 
        total_style_loss = torch.tensor(0.0, device=device)

        for i, (style_gram, mask) in enumerate(zip(styles, masks)):
            region_style_loss = torch.tensor(0.0, device=device)
            masked_x = x * mask.unsqueeze(0)
            with torch.set_grad_enabled(True):
                _, masked_style = net(masked_x, content_list, style_list)
                for j in range(len(masked_style)):
                    region_loss = F.mse_loss(masked_style[j], style_gram[j].detach())
                    region_style_loss = region_style_loss + region_loss
                    total_style_loss = total_style_loss + region_loss
            style_losses.append(region_style_loss.item())

        # 总变差损失
        tv_loss = (torch.sum(torch.abs(x[:, :, :-1] - x[:, :, 1:])) + 
                  torch.sum(torch.abs(x[:, :-1, :] - x[:, 1:, :]))) / (3 * x.shape[2] * x.shape[3])
        loss = parameter_list["content_weight"] * content_loss + \
               parameter_list["style_weight"] * total_style_loss + \
               parameter_list["tv_weight"] * tv_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        if epoch == 0 or epoch % 100 == 99:
            print(f"Epoch {epoch+1}:")
            print(f"Total Loss = {loss.item():.4f}")
            print("----------------------")
            with torch.no_grad():
                img = torch.clamp(x, 0, 1)
                # if dist.get_rank() == 0:  # 只让主进程显示图片
                show_image(img)
    return x


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

def show_image(tensor):
    """
    显示张量图像
    tensor: shape为[1, 3, H, W]的图像张量
    """
    with torch.no_grad():
        # 将张量转换为numpy数组
        image = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # 确保值在[0,1]范围内
        image = np.clip(image, 0, 1)
        
        # 转换为RGB格式显示
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        plt.close()

def save_image(tensor, filename):
    """
    保存张量图像到文件
    tensor: shape为[1, 3, H, W]的图像张量
    filename: 保存的文件名
    """
    # 分离梯度信息
    with torch.no_grad():
        # 将张量转换为numpy数组
        image = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 确保值在[0,1]范围内
        image = np.clip(image, 0, 1)

        # 转换为0-255范围
        image = (image * 255).astype(np.uint8)

        # 转换为BGR格式（OpenCV格式）并保存
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image)


# add by FHT
def resize_with_aspect_ratio(image, max_dim=1025):
    """等比例缩小图像，确保长或宽不超过max_dim"""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h))
    return image


def resize_to_nearest_power_of_two(image, max_dim=1024):
    """调整图像宽和高到小于等于自身且不超过 max_dim 的最大 2 的幂值"""
    h, w = image.shape[:2]

    # 计算宽和高调整后的目标值
    def largest_power_of_two_less_than_or_equal_to(x, limit):
        return min(2 ** int(np.floor(np.log2(x))), limit)

    new_h = largest_power_of_two_less_than_or_equal_to(h, max_dim)
    new_w = largest_power_of_two_less_than_or_equal_to(w, max_dim)

    # 调整图像大小
    return cv2.resize(image, (new_w, new_h))


def style_transform(style_paths: list[str], target_path: str, save_path="result.png"):
    """将若干张图片风格迁移到一张图上
    
    Args:
        style_paths (list[str]): 若干风格图片的路径
        target_path (str): 要迁移到的目标图片
        save_path (str): 生成的图片路径 Default to "result.png"
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载内容图像
    content_img = cv2.imread(target_path)
    original_height, original_width = content_img.shape[:2]  # FHT: 保存原始尺寸
    print(f"\norigin: {content_img.shape}\n")
    # content_img = cv2.resize(content_img, (512, 512))
    content_img = resize_to_nearest_power_of_two(content_img)
    print(f"\n{content_img.shape}\n")
    content_tensor = preprocess_image(content_img).to(device)

    # 加载多个风格图像
    style_tensors = []
    for style_path in style_paths:
        style_img = cv2.imread(style_path)
        # # 确保所有图像大小一致
        # style_img = cv2.resize(style_img, (512, 512))

        # modified by FHT
        # 根据目标图像的大小调整风格图像的大小
        target_height, target_width = content_img.shape[:2]
        style_img = cv2.resize(style_img, (target_width, target_height))
        print(f"\n{style_img.shape}\n")

        style_tensors.append(preprocess_image(style_img).to(device))

    # 设置参数
    parameter_list = {
        "device": device,
        "epoches": 1000,
        "lr": 1e-3,
        "content_weight": 1e-2,  # origin: 1e-2
        "style_weight": 1e7,  # origin: 1e7
        "tv_weight": 1e-3,
        "content_list": [25],
        "style_list": [0, 5, 10, 19, 28]
    }

    # 初始化网络
    # net = transformer(models.vgg19(pretrained=True).features.eval()).to(device)
    net = transformer(models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()).to(device)

    # 使用 DataParallel 包裹模型
    net = nn.DataParallel(net, device_ids=[0, 1]).to(device)

    # 获取内容特征
    content_features, _ = net(content_tensor, parameter_list["content_list"], parameter_list["style_list"])

    # 获取多个风格的 Gram 矩阵
    style_grams = []
    for style_tensor in style_tensors:
        _, style_gram = net(style_tensor, parameter_list["content_list"], parameter_list["style_list"])
        style_grams.append(style_gram)

    x = content_tensor.clone()
    result = train(x, net, parameter_list, content_features, style_grams)

    # FHT: 输出的图像resize为原始内容图像大小
    result = F.interpolate(
        result,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )

    save_image(result, save_path)

# def style_transform(style_paths: list[str], target_path: str, save_path="result.png"):
#     """将若干张图片风格迁移到一张图上

#     Args:
#         style_paths (list[str]): 若干风格图片的路径
#         target_path (str): 要迁移到的目标图片
#         save_path (str): 生成的图片路径 Default to "result.png"
#     """

#     # 1. 设置分布式训练环境
#     dist.init_process_group(backend="nccl", init_method="env://", world_size=2, rank=int(os.environ['RANK']))
#     device = torch.device(f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu")

#     # 2. 加载内容图像
#     content_img = cv2.imread(target_path)
#     content_img = cv2.resize(content_img, (512, 512))
#     content_tensor = preprocess_image(content_img).to(device)

#     # 3. 加载多个风格图像
#     style_tensors = []
#     for style_path in style_paths:
#         style_img = cv2.imread(style_path)
#         style_img = cv2.resize(style_img, (512, 512))
#         style_tensors.append(preprocess_image(style_img).to(device))

#     # 设置参数
#     parameter_list = {
#         "device": device,
#         "epoches": 2000,
#         "lr": 1e-3,
#         "content_weight": 1e-2,
#         "style_weight": 1e7,
#         "tv_weight": 1e-3,
#         "content_list": [25],
#         "style_list": [0, 5, 10, 19, 28]
#     }

#     # 4. 初始化网络
#     net = transformer(models.vgg19(pretrained=True).features.eval()).to(device)

#     # 5. 使用 DistributedDataParallel 包裹模型
#     net = nn.parallel.DistributedDataParallel(net, device_ids=[dist.get_rank()], output_device=dist.get_rank())

#     # 6. 获取内容特征
#     content_features, _ = net(content_tensor, parameter_list["content_list"], parameter_list["style_list"])

#     # 7. 获取多个风格的 Gram 矩阵
#     style_grams = []
#     for style_tensor in style_tensors:
#         _, style_gram = net(style_tensor, parameter_list["content_list"], parameter_list["style_list"])
#         style_grams.append(style_gram)

#     x = content_tensor.clone()

#     # 8. 使用 train 函数进行训练
#     result = train(x, net, parameter_list, content_features, style_grams)

#     # 9. 保存最终结果
#     if dist.get_rank() == 0:  # 只让主进程保存图片
#         save_image(result, save_path)

#     # 10. 清理分布式训练环境
#     dist.destroy_process_group()


if __name__ == "__main__":
    style_transform(
        style_paths=["../data/Monet/sunrise.png", "../data/style/style6.png", "../data/VanGogh/starry.jpg"],
        target_path="../data/content/roadview.jpg",
    )
