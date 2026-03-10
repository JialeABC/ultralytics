import xml.etree.ElementTree as ET
import os


def convert_xml_to_yolo(xml_path, txt_path, class_map):
    """
    将包含 polygon 的 VOC XML 转换为 YOLO 格式的 TXT (基于外接矩形框)
    :param xml_path: 输入 XML 路径
    :param txt_path: 输出 TXT 路径
    :param class_map: 类别名称到 ID 的映射字典 {'car': 0, 'person': 1, ...}
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. 获取图片尺寸
    size = root.find('size')
    if size is None:
        print(f"警告: {xml_path} 中未找到 <size> 标签")
        return
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    if img_width == 0 or img_height == 0:
        print(f"错误: {xml_path} 图片尺寸为 0")
        return

    lines = []

    # 2. 遍历所有物体
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in class_map:
            print(f"警告: 未知类别 '{name}'，已跳过。请在 class_map 中添加。")
            continue

        cls_id = class_map[name]

        # 检查是否有 polygon 数据
        polygon = obj.find('polygon')
        if polygon is not None:
            # 提取所有坐标点
            x_coords = []
            y_coords = []

            # 遍历 polygon 下的所有 xN, yN 标签
            # 注意：XML 中可能是 x1, x2, x3... 顺序不固定，需要动态获取
            for child in polygon:
                tag = child.tag
                val = float(child.text)
                if tag.startswith('x'):
                    x_coords.append(val)
                elif tag.startswith('y'):
                    y_coords.append(val)

            if len(x_coords) < 3 or len(y_coords) < 3:
                continue

            # 计算外接矩形 (Bounding Box)
            xmin = min(x_coords)
            xmax = max(x_coords)
            ymin = min(y_coords)
            ymax = max(y_coords)

            # 转换为 YOLO 格式 (归一化中心点)
            # x_center = (xmin + xmax) / 2 / width
            # y_center = (ymin + ymax) / 2 / height
            # w = (xmax - xmin) / width
            # h = (ymax - ymin) / height

            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            w = (xmax - xmin) / float(img_width)
            h = (ymax - ymin) / float(img_height)

            # 格式化输出 (保留6位小数)
            line = f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            lines.append(line)

        else:
            # 如果没有 polygon，检查是否有传统的 bndbox (以防万一)
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = float(bndbox.find('xmin').text)
                xmax = float(bndbox.find('xmax').text)
                ymin = float(bndbox.find('ymin').text)
                ymax = float(bndbox.find('ymax').text)

                x_center = (xmin + xmax) / 2.0 / img_width
                y_center = (ymin + ymax) / 2.0 / img_height
                w = (xmax - xmin) / float(img_width)
                h = (ymax - ymin) / float(img_height)

                line = f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                lines.append(line)

    # 3. 写入文件
    with open(txt_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

    print(f"转换成功: {os.path.basename(xml_path)} -> {len(lines)} 个目标")


# ================= 配置区域 =================
if __name__ == "__main__":
    # 1. 定义类别映射 (根据你的 XML 内容修改)
    # 你的 XML 里只有 'car'，假设它是第 0 类。如果有其他类，请继续添加
    # VisDrone 示例映射:
    class_mapping = {
        "car":0,
        "truck":1,
        "bus":2,
        "van":3,
        "freight_car":4
    }

    # 2. 设置单个文件测试路径 (或者你可以写循环批量处理)
    # 请将这里改为你实际的 XML 文件路径
    xml_file = r"D:/A_my_study/visdrone/test/testlabelr/00078.xml"
    # 输出的 txt 路径 (通常和圖片同名，放在 labels 文件夹)
    txt_file = r"D:/A_my_study/visdrone/test/vis/00078.txt"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)

    # 执行转换
    if os.path.exists(xml_file):
        convert_xml_to_yolo(xml_file, txt_file, class_mapping)
    else:
        print(f"错误：找不到文件 {xml_file}")
        print("请检查路径是否正确，注意 XML 后缀名是否为 .xml (你提供的文本是 XML 内容，但文件可能是 .xml)")

    # ================= 批量处理示例 (如果需要处理整个文件夹) =================
    # input_folder = r"C:\Users\linkdata\Desktop\1\拉框\0620\5"
    # output_folder = r"D:\A_my_study\visdrone\yolov5_r\labels\val"
    # os.makedirs(output_folder, exist_ok=True)
    #
    # for filename in os.listdir(input_folder):
    #     if filename.endswith(".xml"):
    #         xml_path = os.path.join(input_folder, filename)
    #         txt_name = os.path.splitext(filename)[0] + ".txt"
    #         txt_path = os.path.join(output_folder, txt_name)
    #         convert_xml_to_yolo(xml_path, txt_path, class_mapping)