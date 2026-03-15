import pandas as pd
import matplotlib.pyplot as plt
import os

# ==================== 全局配置（解决中文显示+图表样式） ====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适配中文（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.facecolor'] = 'white'  # 画布背景为白色


def plot_Triplet_loss_curve(csv_path, encoding='utf-8'):
    """
    绘制train/Triplet_loss随epoch变化的曲线（无图例）
    :param csv_path: CSV文件路径（绝对/相对路径）
    :param encoding: 文件编码（默认utf-8，乱码时改gbk）
    """
    # 1. 校验文件是否存在
    if not os.path.exists(csv_path):
        print(f"❌ 错误：文件 {csv_path} 不存在！")
        return

    try:
        # 2. 读取CSV数据
        df = pd.read_csv(csv_path, encoding=encoding)

        # 3. 校验列名是否存在（仅校验核心列：epoch、train/Triplet_loss）
        required_cols = ['epoch', 'train/Triplet_loss']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 错误：CSV中缺少列 {missing_cols}！")
            print(f"当前CSV包含的列：{list(df.columns)}")
            return

        # 4. 数据清洗（去除空值、异常值）
        df_clean = df[required_cols].dropna()  # 删除空值行
        # 过滤loss异常值（可选，根据你的数据范围调整）
        df_clean = df_clean[df_clean['train/Triplet_loss'] > 0]  # _loss应为正数

        if len(df_clean) == 0:
            print("❌ 错误：有效数据为空！")
            return

        # 5. 按epoch排序（保证曲线连贯）
        df_clean = df_clean.sort_values(by='epoch').reset_index(drop=True)

        # 6. 创建画布（仅单Y轴：Loss）
        fig, ax1 = plt.subplots(figsize=(14, 8))  # 画布尺寸放大

        # 6.1 绘制train/Triplet_loss曲线（无图例）
        color1 = '#2E86AB'
        # 横坐标标签：字号增大到18，间距增大
        ax1.set_xlabel('Epoch', fontsize=18, labelpad=15, fontweight='bold')
        # 纵坐标标签：字号增大到18，间距增大，加粗
        ax1.set_ylabel('Loss', fontsize=18, color=color1, labelpad=15, fontweight='bold')
        ax1.plot(
            df_clean['epoch'],
            df_clean['train/Triplet_loss'],
            marker='o',  # 标记改为圆形，更醒目
            markersize=8,  # 标记尺寸从4→8
            linestyle='-',  # 实线
            linewidth=3,  # 曲线宽度从2→3
            color=color1
        )
        # 坐标轴刻度值字号增大：从默认→16
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)  # 网格线加粗

        # 7. 图表样式优化
        # 标题字号从16→20，加粗，间距增大
        ax1.set_title('Triplet_loss', fontsize=20, pad=25, fontweight='bold')

        # 调整布局（防止标签重叠）
        plt.tight_layout()

        # 8. 显示/保存图表
        plt.show()
        # 可选：保存高清图片（建议保留，论文用300DPI高清图）
        # fig.savefig('Triplet_loss_curve.png', dpi=300, bTriplet_inches='tight')

    except UnicodeDecodeError:
        print(f"❌ 错误：文件编码不是 {encoding}！请尝试修改encoding为'gbk'")
    except Exception as e:
        print(f"❌ 程序执行出错：{str(e)}")


# ==================== 示例调用 ====================
if __name__ == '__main__':
    # 替换为你的CSV文件路径
    CSV_PATH = "D:/Deeplearning_code/yolov8/ultralytics/runs/detect/only_car/your_need_results.csv"
    ENCODING = "utf-8"  # 乱码时改为"gbk"

    # 调用绘图函数
    plot_Triplet_loss_curve(CSV_PATH, ENCODING)