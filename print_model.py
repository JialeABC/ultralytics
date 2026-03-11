import torch

# 加载文件内容
data = torch.load('D:/Deeplearning_code/yolov8/ultralytics/runs/detect/best.pt', map_location='cpu')

# 情况 A: 如果保存的是整个模型对象 (较少见，但可以直接打印)
if hasattr(data, 'state_dict'):
    print(data)

# 情况 B: 如果保存的是 state_dict (最常见)，打印所有的层参数名
elif isinstance(data, dict):
    # 尝试找 'model' 键 (YOLO等常见格式)
    if 'model' in data and hasattr(data['model'], 'state_dict'):
        print(data['model'])
    else:
        # 直接打印所有参数的名字，你能看到类似 "layer1.conv.weight" 的名字
        print("检测到的层参数名列表：")
        for key in data.keys():
            print(f"- {key} (形状: {data[key].shape})")
else:
    print("文件格式未知，无法直接解析结构。")