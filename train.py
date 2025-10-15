from ultralytics import YOLO


if __name__ == '__main__':
    # model = YOLO("yolov8n.pt")  # 从头开始构建新模型
    model = YOLO("yolov8n.pt")
    results = model.train(data="D:/Deeplearning_code/ultralytics-main/ultralytics/cfg/datasets/coco.yaml", epochs=100,
                          imgsz=1600)