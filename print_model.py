from ultralytics import YOLO

# 直接加载模型
model = YOLO('D:/Deeplearning_code/yolov8/ultralytics/runs/detect/train8/weights/best.pt')

# 打印模型信息（包括层数、参数数量、形状等）
model.info()

# 或者查看具体的模型架构
print(model.model)