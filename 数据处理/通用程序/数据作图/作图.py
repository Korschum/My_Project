import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# ================= 用户配置区域 (在此处修改) =================

# 1. 数据文件路径列表
DATA_FILE_LIST = [
    r'/Users/vassago/Desktop/资料/BY/02 动态范围/实验数据分析/20260414/Laser0_summary_median_data.csv',
    r'/Users/vassago/Desktop/资料/BY/02 动态范围/实验数据分析/20260414/Laser1_summary_median_data.csv',
    # 在此处添加更多文件路径...
]

# 2. 输出图片保存路径
OUTPUT_IMAGE_PATH = r'/Users/vassago/Desktop/资料/BY/02 动态范围/实验数据分析/20260414/Lasertot_plot_result_fit.png'

# 3. X、Y轴变量选择及绘图标签
# 根据你的描述，这里调整为：X轴是P_opt (对数)，Y轴是Median_Current (线性)
X_AXIS_COL = 'P_opt'                # 数据中的列名
Y_AXIS_COL = 'Median_Current'       # 数据中的列名
X_AXIS_LABEL = 'Light Intensity ($W/cm^2$)'  # 绘图时X轴显示的标签
Y_AXIS_LABEL = '$I_{sd} (A)$'       # 绘图时Y轴显示的标签 (支持LaTeX)

# 4. 数据筛选条件 (字典形式，键为列名，值为要筛选的值列表)
FILTER_CONDITIONS = {'Vg': [0.7], 'Vd': [0.4, 0.6]}

# 5. 用于分组、着色和标注的列
LABEL_COL = 'Vd'

# 6. 对数坐标轴设置
ENABLE_X_LOG = True   # 开启X轴对数坐标
ENABLE_Y_LOG = False  # 关闭Y轴对数坐标 (根据你的描述)

# 7. 零值/无效数据标记样式
ZERO_MARKER_STYLE = True

# 8. 线性拟合设置
ENABLE_LINEAR_FIT = True
FIT_LINE_STYLE = '--'
FIT_LINE_WIDTH = 2.0
SHOW_FIT_EQ_ON_LEGEND = False
SHOW_FIT_EQ_ON_PLOT = False

# 9. 拟合数据范围限制 (注意：这里的范围是指原始数据的范围)
# 对于对数坐标拟合，范围限制依然基于原始值
FIT_X_RANGE = [1e-7, 1]
FIT_Y_RANGE = None

# 10. 数据点连接设置
CONNECT_POINTS = False

# ================= 核心逻辑区域 (无需修改) =================

def load_and_filter_data(file_list, filter_conditions):
    """读取多个CSV文件并根据条件筛选数据（增加单位转换）"""
    all_data = []
    for file_path in file_list:
        if os.path.exists(file_path):
            try:
                # 1. 读取数据
                df = pd.read_csv(file_path)
                
                # 2. 定义一个辅助函数来转换带有单位的字符串
                def convert_unit(val):
                    if pd.isna(val): # 处理空值
                        return np.nan
                    val_str = str(val)
                    if 'u' in val_str: # 微单位 10^-6
                        return float(val_str.replace('u', '')) * 1e-6
                    elif 'm' in val_str: # 毫单位 10^-3
                        return float(val_str.replace('m', '')) * 1e-3
                    else: # 纯数字或其他
                        return float(val_str)
                
                # 3. 转换格式
                if Y_AXIS_COL in df.columns:
                    df[Y_AXIS_COL] = df[Y_AXIS_COL].apply(convert_unit)
                if X_AXIS_COL in df.columns:
                    df[X_AXIS_COL] = df[X_AXIS_COL].apply(convert_unit)
                
                all_data.append(df)
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
        else:
            print(f"文件不存在: {file_path}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 5. 原有的筛选逻辑
    for col, values in filter_conditions.items():
        if col in combined_df.columns:
            # 注意：这里的 values (如 [1, 0.20]) 是数字，如果列被转成了字符串需要处理
            # 但通常 Slit_in/Slit_out 是纯数字，所以一般不需要改
            combined_df = combined_df[combined_df[col].isin(values)]
    
    return combined_df

def perform_linear_fit(x_data, y_data, x_range, y_range, enable_x_log, enable_y_log):
    """
    执行线性拟合。
    如果开启了对数坐标，会对相应数据进行对数变换后再拟合，
    但返回的坐标值仍然是原始数值，以便直接在图上绘制。
    """
    # 1. 数据清洗：移除NaN和无穷值
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    # 额外处理：对数坐标下，数值必须大于0
    if enable_x_log:
        valid_mask &= (x_data > 0)
    if enable_y_log:
        valid_mask &= (y_data > 0)
        
    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]
    
    if len(x_clean) < 2:
        return None, None, None

    # 2. 应用拟合范围限制 (基于原始数值)
    fit_mask = np.ones(len(x_clean), dtype=bool)
    if x_range is not None:
        x_min, x_max = x_range
        fit_mask &= (x_clean >= x_min) & (x_clean <= x_max)
    if y_range is not None:
        y_min, y_max = y_range
        fit_mask &= (y_clean >= y_min) & (y_clean <= y_max)
        
    x_fit_input = x_clean[fit_mask]
    y_fit_input = y_clean[fit_mask]

    if len(x_fit_input) < 2:
        return None, None, None

    # 3. 准备拟合数据 (根据坐标轴类型进行变换)
    x_for_fit = x_fit_input
    y_for_fit = y_fit_input
    
    if enable_x_log:
        x_for_fit = np.log10(x_fit_input)
    if enable_y_log:
        y_for_fit = np.log10(y_fit_input)

    # 4. 执行线性回归
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_for_fit, y_for_fit)
        
        # 生成用于绘制拟合线的X值 (在原始数据范围内均匀分布)
        x_line_original = np.linspace(x_fit_input.min(), x_fit_input.max(), 100)
        
        # 计算对应的Y值
        # 我们需要根据拟合的模型反推回原始Y值
        y_line_original = []
        
        for x_val in x_line_original:
            # 步骤A: 将当前的X值转换为拟合时使用的形式
            x_val_transformed = np.log10(x_val) if enable_x_log else x_val
            
            # 步骤B: 计算拟合后的Y值 (拟合空间)
            y_val_transformed = slope * x_val_transformed + intercept
            
            # 步骤C: 将Y值转换回原始空间
            if enable_y_log:
                y_val_original = 10 ** y_val_transformed
            else:
                y_val_original = y_val_transformed
            
            y_line_original.append(y_val_original)
            
        y_line_original = np.array(y_line_original)

        # 构造方程字符串 (简化版，仅显示斜率和截距，实际物理意义取决于变换)
        # 这里为了通用性，不写复杂的数学公式，只显示拟合优度或简单标识
        equation_str = f'Fit (Log Space): slope={slope:.2f}'
        
        return x_line_original, y_line_original, equation_str
        
    except Exception as e:
        print(f"拟合出错: {e}")
        return None, None, None

def plot_data(df, x_col, y_col, label_col, x_label, y_label, output_path,
              enable_x_log, enable_y_log, zero_marker, enable_fit,
              fit_style, fit_width, show_fit_eq_legend, show_fit_eq_plot,
              fit_x_range, fit_y_range, connect_points):
    """主绘图函数"""
    if df.empty:
        print("没有数据可绘制。")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = df[label_col].unique()
    colormap = plt.cm.coolwarm                                             # 颜色————————————   
    colors = colormap(np.linspace(0, 1, len(unique_labels)))
    
    for i, label_val in enumerate(sorted(unique_labels)):
        group = df[df[label_col] == label_val].sort_values(by=x_col)
        x_data = group[x_col].to_numpy()
        y_data = group[y_col].to_numpy()
        
        color = colors[i]
        label_name = f'{label_col}={label_val}'
        
        # --- 处理对数坐标下的零值/负值 ---
        x_for_plot, y_for_plot = x_data.copy(), y_data.copy()
        x_zero_indices, y_zero_indices = [], []
        
        if enable_x_log:
            invalid_x = x_data <= 0
            x_for_plot[invalid_x] = np.nan
            x_zero_indices = np.where(invalid_x)[0]
        if enable_y_log:
            invalid_y = y_data <= 0
            y_for_plot[invalid_y] = np.nan
            y_zero_indices = np.where(invalid_y)[0]

        # --- 绘制数据点 ---
        scatter = ax.scatter(x_for_plot, y_for_plot, color=color, label=label_name, zorder=3)
        
        # --- 绘制零值标记 ---
        if zero_marker:
            if enable_x_log:
                x_min_plot = np.nanmin(x_for_plot) if np.any(np.isfinite(x_for_plot)) else 1e-10
            else:
                x_min_plot = ax.get_xlim()[0]
            
            if enable_y_log:
                y_min_plot = np.nanmin(y_for_plot) if np.any(np.isfinite(y_for_plot)) else 1e-10
            else:
                y_min_plot = ax.get_ylim()[0]

            for idx in x_zero_indices:
                y_val = y_data[idx]
                if np.isscalar(y_val) and np.isfinite(y_val):
                    y_marker_pos = y_val
                    if enable_y_log and y_val <= 0:
                        y_marker_pos = y_min_plot * 0.9
                    ax.plot([x_min_plot * 0.9, x_min_plot * 0.95], [y_marker_pos, y_marker_pos], color=color, linewidth=2, zorder=4)
            
            for idx in y_zero_indices:
                x_val = x_data[idx]
                if np.isscalar(x_val) and np.isfinite(x_val):
                    x_marker_pos = x_val
                    if enable_x_log and x_val <= 0:
                        x_marker_pos = x_min_plot * 0.9
                    ax.plot([x_marker_pos, x_marker_pos], [y_min_plot * 0.9, y_min_plot * 0.95], color=color, linewidth=2, zorder=4)

        # --- 连接数据点 ---
        if connect_points:
            ax.plot(x_for_plot, y_for_plot, color=color, linestyle='-', linewidth=1, alpha=0.6, zorder=2)

        # --- 线性拟合 ---
        if enable_fit:
            # 传入坐标轴设置，以便函数内部进行对数变换
            x_line, y_line, eq_str = perform_linear_fit(x_data, y_data, fit_x_range, fit_y_range, enable_x_log, enable_y_log)
            if x_line is not None:
                ax.plot(x_line, y_line, color=color, linestyle=fit_style, linewidth=fit_width, zorder=1)
                if show_fit_eq_plot:
                    ax.text(0.05, 0.95 - i*0.05, eq_str, transform=ax.transAxes, color=color, verticalalignment='top')
                
                if show_fit_eq_legend:
                    handles, labels = ax.get_legend_handles_labels()
                    for j, l in enumerate(labels):
                        if l == label_name:
                            handles[j].set_label(f'{label_name}\n{eq_str}')
                            break

    # --- 设置坐标轴 ---
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{y_label} vs {x_label}', fontsize=14)
    
    if enable_x_log:
        ax.set_xscale('log')
    if enable_y_log:
        ax.set_yscale('log')
        
    ax.grid(True, which='both', linestyle='--', alpha=0.5, zorder=0)
    ax.legend(title=f'Grouped by {label_col}', fontsize=10)
    
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已成功保存至: {output_path}")
    except Exception as e:
        print(f"保存图片时出错: {e}")
    plt.close()

# ================= 程序入口 =================
if __name__ == "__main__":
    print("正在加载并筛选数据...")
    data_df = load_and_filter_data(DATA_FILE_LIST, FILTER_CONDITIONS)
    
    print("正在绘制图表...")
    plot_data(
        df=data_df,
        x_col=X_AXIS_COL,
        y_col=Y_AXIS_COL,
        label_col=LABEL_COL,
        x_label=X_AXIS_LABEL,
        y_label=Y_AXIS_LABEL,
        output_path=OUTPUT_IMAGE_PATH,
        enable_x_log=ENABLE_X_LOG,
        enable_y_log=ENABLE_Y_LOG,
        zero_marker=ZERO_MARKER_STYLE,
        enable_fit=ENABLE_LINEAR_FIT,
        fit_style=FIT_LINE_STYLE,
        fit_width=FIT_LINE_WIDTH,
        show_fit_eq_legend=SHOW_FIT_EQ_ON_LEGEND,
        show_fit_eq_plot=SHOW_FIT_EQ_ON_PLOT,
        fit_x_range=FIT_X_RANGE,
        fit_y_range=FIT_Y_RANGE,
        connect_points=CONNECT_POINTS
    )
    print("程序执行完毕。")