import math
import xml.etree.ElementTree as ET
import os


# 计算两点之间的欧氏距离
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 计算四个角点的中心点
def get_center(x1, y1, x2, y2, x3, y3, x4, y4):
    x_center = (x1 + x2 + x3 + x4) / 4.0
    y_center = (y1 + y2 + y3 + y4) / 4.0
    return x_center, y_center


# 计算角度（相对于水平线的角度）
def get_angle(x1, y1, x3, y3):
    dx = x3 - x1
    dy = y3 - y1
    angle = math.degrees(math.atan2(dy, dx))  # 返回角度值，单位：度
    return angle


# 将坐标转换为YOLO格式
def convert_rotated_bbox_to_yolo(xml_file, image_width, image_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取图像尺寸
    image_width = int(root.find('.//size/width').text)
    image_height = int(root.find('.//size/height').text)

    yolo_annotations = []

    for obj in root.findall('object'):
        # 获取目标的类别
        class_name = obj.find('name').text
        class_id = 0  # 假设只有一个类别 "car"，类ID为0，如果有更多类，请修改

        # 获取四个角点坐标
        polygon = obj.find('polygon')
        x1 = int(polygon.find('x1').text)
        y1 = int(polygon.find('y1').text)
        x2 = int(polygon.find('x2').text)
        y2 = int(polygon.find('y2').text)
        x3 = int(polygon.find('x3').text)
        y3 = int(polygon.find('y3').text)
        x4 = int(polygon.find('x4').text)
        y4 = int(polygon.find('y4').text)

        # 计算中心点
        x_center, y_center = get_center(x1, y1, x2, y2, x3, y3, x4, y4)

        # 计算宽度和高度
        width = (distance(x1, y1, x3, y3) + distance(x2, y2, x4, y4)) / 2.0 / image_width
        height = (distance(x1, y1, x4, y4) + distance(x2, y2, x3, y3)) / 2.0 / image_height

        # 计算旋转角度
        angle = get_angle(x1, y1, x3, y3)

        # 转换为 YOLO 格式
        x_center /= image_width
        y_center /= image_height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height} {angle}")

    return yolo_annotations


# 保存 YOLO 格式到 TXT 文件
def save_yolo_format(txt_file, yolo_annotations):
    with open(txt_file, 'w') as f:
        for annotation in yolo_annotations:
            f.write(f"{annotation}\n")


# 主程序
def convert_xml_to_yolo(xml_file, image_size=(840, 712)):
    # 获取 YOLO 格式标注
    yolo_annotations = convert_rotated_bbox_to_yolo(xml_file, image_size[0], image_size[1])

    # 保存为TXT文件
    txt_file = xml_file.replace(".xml", ".txt")
    save_yolo_format(txt_file, yolo_annotations)
    print(f"YOLO 格式标注已保存：{txt_file}")


# 示例：将XML文件转换为YOLO格式
label_folder = "D:/A_my_study/visdrone/test/test/testlabel"
new_folder = "D:/A_my_study/visdrone/folder/"
for i in os.listdir(label_folder):
    name , ext = os.path.splitext(i)
    old_filename = os.path.join(label_folder, i)
    new_filename = new_folder + name + ".txt"
    convert_xml_to_yolo(old_filename)
