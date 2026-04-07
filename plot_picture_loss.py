import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager

# ==================== 全局字体配置 ====================
# 1. 设置中文字体为宋体 (SimSun)
# Windows 系统通常自带 'SimSun'，Mac/Linux 可能需要安装或改用 'STSong'
plt.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']  # 优先宋体

# 2. 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 3. 画布背景
plt.rcParams['figure.facecolor'] = 'white'

# 【重要】为了强制英文使用 Times New Roman，我们需要在绘图时显式指定
# 因为全局设置很难同时完美区分中英文字体（除非使用复杂的字体回退机制）
# 这里定义一个通用的字参数字典，方便复用
FONT_CONFIG = {
    'family': 'sans-serif',  # 基础家族
    'weight': 'bold'  # 默认加粗
}


def plot_Triplet_loss_curve(csv_path, encoding='utf-8'):
    """
    绘制 train/Triplet_loss 曲线：中文宋体，英文 Times New Roman
    """
    if not os.path.exists(csv_path):
        print(f"❌ 错误：文件 {csv_path} 不存在！")
        return

    try:
        df = pd.read_csv(csv_path, encoding=encoding)

        required_cols = ['epoch', 'train/Triplet_loss']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 错误：CSV 中缺少列 {missing_cols}！")
            print(f"当前 CSV 包含的列：{list(df.columns)}")
            return

        df_clean = df[required_cols].dropna()
        df_clean = df_clean[df_clean['train/Triplet_loss'] > 0]

        if len(df_clean) == 0:
            print("❌ 错误：有效数据为空！")
            return

        df_clean = df_clean.sort_values(by='epoch').reset_index(drop=True)

        fig, ax1 = plt.subplots(figsize=(14, 8))

        color1 = '#2E86AB'

        # --- 设置坐标轴标签 (混合字体处理) ---
        # 方法：使用 fontproperties 参数指定具体字体文件，或利用名称
        # 对于中文标签，我们主要依赖全局的 SimSun
        # 对于纯英文标签，我们强制指定 'Times New Roman'

        # X 轴标签 ("Epoch" 是英文)
        ax1.set_xlabel(
            'Epoch',
            fontsize=30,
            labelpad=15,
            fontweight='bold',
            fontname='Times New Roman'  # 强制英文字体
        )

        # Y 轴标签 ("Loss" 是英文)
        ax1.set_ylabel(
            'Loss',
            fontsize=30,
            color="black",
            labelpad=15,
            fontweight='bold',
            fontname='Times New Roman'  # 强制英文字体
        )

        # 绘制曲线
        ax1.plot(
            df_clean['epoch'],
            df_clean['train/Triplet_loss'],
            marker='o',
            markersize=8,
            linestyle='-',
            linewidth=3,
            color=color1
        )

        # 设置刻度值字体
        # 刻度值通常是数字，我们希望数字也是 Times New Roman
        for label in ax1.get_xticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(25)

        for label in ax1.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(25)
            label.set_color("black")

        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

        # --- 设置标题 ---
        # 如果你的标题包含中文，例如 "分类损失曲线"，全局的 SimSun 会生效
        # 如果标题是纯英文 "Triplet_loss"，fontname 会覆盖为 Times New Roman
        ax1.set_title(
            'Triplet_loss',  # 这里如果是中文，会自动用宋体；如果是英文，建议下面加 fontname
            fontsize=30,
            pad=25,
            fontweight='bold',
            fontname='Times New Roman'  # 确保英文标题也是 TN
        )

        plt.tight_layout()
        plt.show()

        # 保存示例 (如果需要)
        # fig.savefig('Triplet_loss_curve.png', dpi=300, bTriplet_inches='tight')

    except UnicodeDecodeError:
        print(f"❌ 错误：文件编码不是 {encoding}！请尝试修改 encoding 为 'gbk'")
    except Exception as e:
        print(f"❌ 程序执行出错：{str(e)}")


if __name__ == '__main__':
    CSV_PATH = "D:/Deeplearning_code/yolov8/ultralytics/runs/detect/only_car/your_need_results.csv"
    ENCODING = "utf-8"

    # 检查系统中是否有 SimSun (防止非 Windows 系统报错)
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    if 'SimSun' not in available_fonts:
        print("⚠️ 警告：系统中未检测到 'SimSun' (宋体)。")
        print("   Windows 用户通常无需担心。")
        print("   Mac/Linux 用户请安装宋体，或将代码中的 'SimSun' 改为 'STSong' 或 'WenQuanYi Micro Hei'。")
        # 临时降级处理，防止崩溃
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    plot_Triplet_loss_curve(CSV_PATH, ENCODING)