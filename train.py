from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO("D:/Deeplearning_code/yolov8/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml")  # 从头开始构建新模型
    # model = YOLO("D:/Deeplearning_code/yolov8/ultralytics/runs/detect/train2/weights/best.pt")
    results = model.train(data="D:/Deeplearning_code/yolov8/ultralytics/ultralytics/cfg/datasets/coco8.yaml", epochs=100,
                          imgsz=640,patience=10,resume=True)
    # results = model.val(data="D:/Deeplearning_code/yolov8/ultralytics/ultralytics/cfg/datasets/coco8.yaml")
    #
    # results = model.predict(source='D:/A_my_study/visdrone/yolov5_r/val',save=True)