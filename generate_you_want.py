import cv2
import os
import random


def quick_vis(img_path, txt_path, out_path, classes):
    img = cv2.imread(img_path)
    if img is None: return print("图片读取失败")

    h, w, _ = img.shape
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
              range(len(classes) + 10)]

    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                data = list(map(float, line.split()))
                if len(data) < 5: continue

                cls = int(data[0])
                # 还原坐标
                x1 = int((data[1] - data[3] / 2) * w)
                y1 = int((data[2] - data[4] / 2) * h)
                x2 = int((data[1] + data[3] / 2) * w)
                y2 = int((data[2] + data[4] / 2) * h)

                # 限制边界
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                color = colors[cls % len(colors)]
                label = classes[cls] if cls < len(classes) else str(cls)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(out_path, img)
    print(f"完成：{out_path}")


# 运行配置
if __name__ == "__main__":
    # 修改这里
    src_img = r"D:/A_my_study/visdrone/test/testimgr/00001.jpg"
    src_txt = r"D:/A_my_study/visdrone/test/vis/00001.txt"  # 确认标签路径
    dst_img = r"D:/A_my_study/visdrone/test/vis/00001.jpg"

    cls_names = ["car", "truck", "bus", "van", "freight_car"]

    quick_vis(src_img, src_txt, dst_img, cls_names)