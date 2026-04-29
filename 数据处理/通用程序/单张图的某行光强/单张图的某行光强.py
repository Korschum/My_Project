import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_5x5_average_rgb(img, center_y, center_x):
    """
    获取以 (center_x, center_y) 为中心的 5x5 区域内 RGB 的平均值
    """
    height, width, _ = img.shape
    
    # 1. 计算 5x5 区域的边界
    # 半径为 2 (因为 2 + 1 + 2 = 5)
    radius = 2
    y_start = max(0, center_y - radius)
    y_end = min(height, center_y + radius + 1)
    x_start = max(0, center_x - radius)
    x_end = min(width, center_x + radius + 1)
    
    # 2. 提取 5x5 区域 (切片操作)
    # 注意：OpenCV 数组索引顺序是 [y, x]
    region = img[y_start:y_end, x_start:x_end]
    
    # 3. 计算该区域内每个通道的平均值
    # axis=(0, 1) 表示在高度(0)和宽度(1)这两个维度上求平均
    avg_b, avg_g, avg_r = np.mean(region, axis=(0, 1))
    
    return int(avg_r), int(avg_g), int(avg_b)

def plot_rgb_row(image_path, row_y):
    # 1. 读取图像
    # 注意：OpenCV 默认以 BGR 格式读取图像，而不是 RGB
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"错误：无法找到或打开图片 '{image_path}'")
        return

    # 获取图像尺寸，防止输入的行号超出范围
    height, width, _ = img.shape
    if row_y < 0 or row_y >= height:
        print(f"错误：行号 {row_y} 超出范围 (0 - {height-1})")
        return

    # 2. 提取指定行的 R, G, B 数据
    # OpenCV 通道顺序为 B(0), G(1), R(2)
    b_channel = img[row_y, :, 0]
    g_channel = img[row_y, :, 1]
    r_channel = img[row_y, :, 2]
    
    radius = 100
    # 确保不越界
    y_start = max(0, row_y - radius)
    y_end = min(height, row_y + radius + 1)

    # 提取 5 行高度的数据
    region_5_rows = img[y_start:y_end, :, :]

    # 计算垂直方向（5行）的平均值，结果变回 1 行高度
    # axis=0 表示对行求平均
    avg_img_row = np.mean(region_5_rows, axis=0)

    # 分离通道
    b_channel_m = avg_img_row[:, 0]
    g_channel_m = avg_img_row[:, 1]
    r_channel_m = avg_img_row[:, 2]


    # 创建 X 轴坐标 (0 到 图像宽度)
    x = np.arange(width)

    # 3. 绘制图表
    # 设置画布大小 (宽, 高)
    plt.figure(figsize=(15, 10))

    # --- 子图 1: 显示原图并标记出选中的行 ---
    plt.subplot(2, 1, 1)
    # 将 BGR 转回 RGB 以便正确显示颜色
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    # 在图上画出红线标记位置
    plt.plot([0, width], [row_y, row_y], color='red', linewidth=2, label=f'Row {row_y}')
    plt.title(f'Original Image (Red line indicates Row {row_y})')
    plt.axis('off') # 关闭坐标轴显示
    plt.legend()

    # --- 子图 2: 绘制 RGB 数值折线图 ---
    plt.subplot(2, 1, 2)
    plt.plot(x, r_channel_m, color='red', label='Red Channel', alpha=0.7)
    plt.plot(x, g_channel_m, color='green', label='Green Channel', alpha=0.7)
    plt.plot(x, b_channel_m, color='blue', label='Blue Channel', alpha=0.7)
    
    plt.title(f'RGB Values Along Row {row_y}')
    plt.xlabel('Pixel Position (X-axis)')
    plt.ylabel('Intensity Value (0-255)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6) # 显示网格

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

# ================= 使用示例 =================
# 请将此处替换为你自己的图片路径
image_file = '/Users/vassago/Desktop/资料/BY/02 动态范围/实验数据分析/20260419/DCIM/Pictures/2_6.jpg' 
# image_file = '/Users/vassago/Desktop/资料/BY/02 动态范围/实验数据分析/20260419/DCIM/Pictures/1.jpg'
# image_file = '/Users/vassago/Desktop/资料/BY/02 动态范围/实验数据分析/20260419/DCIM/Pictures/2_1.jpg'

# 指定你想分析的行号 (例如第 100 行)
target_row = 1000

# 运行函数
plot_rgb_row(image_file, target_row)