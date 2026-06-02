import os
import json
import argparse
import xml.etree.ElementTree as ET
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VOC2007 dataset to COCO format")

    parser.add_argument(
        "--voc-root",
        type=str,
        default="D:/A_my_study/FLIR/VOC2007_r/",
        help="Path to VOC2007 directory, e.g. D:/dataset/VOC2007"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name, e.g. train, val, test"
    )

    parser.add_argument(
        "--save-json",
        type=str,
        default="D:/A_my_study/FLIR/coco/annotations/instances_train2017.json",
        help="Output COCO json path, e.g. D:/dataset/annotations/train.json"
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=["vehicle"],
        nargs="+",
        help="Class names, e.g. --classes person car bus"
    )

    parser.add_argument(
        "--ignore-difficult",
        action="store_true",
        help="Ignore objects with difficult=1"
    )

    return parser.parse_args()


def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def voc_xml_to_coco(voc_root, split, save_json, classes, ignore_difficult=False):
    ann_dir = os.path.join(voc_root, "Annotations")
    img_dir = os.path.join(voc_root, "JPEGImages")
    split_file = os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"Annotations directory not found: {ann_dir}")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"JPEGImages directory not found: {img_dir}")

    class_to_id = {cls_name: idx + 1 for idx, cls_name in enumerate(classes)}

    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for cls_name, cls_id in class_to_id.items():
        coco["categories"].append({
            "id": cls_id,
            "name": cls_name,
            "supercategory": "none"
        })

    with open(split_file, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f.readlines() if line.strip()]

    ann_id = 1
    img_id = 1

    skipped_unknown_classes = set()
    skipped_missing_files = []

    for image_name in image_ids:
        xml_path = os.path.join(ann_dir, image_name + ".xml")

        jpg_path = os.path.join(img_dir, image_name + ".jpg")
        jpeg_path = os.path.join(img_dir, image_name + ".jpeg")
        png_path = os.path.join(img_dir, image_name + ".png")

        if os.path.exists(jpg_path):
            image_path = jpg_path
            file_name = image_name + ".jpg"
        elif os.path.exists(jpeg_path):
            image_path = jpeg_path
            file_name = image_name + ".jpeg"
        elif os.path.exists(png_path):
            image_path = png_path
            file_name = image_name + ".png"
        else:
            skipped_missing_files.append(image_name)
            continue

        if not os.path.exists(xml_path):
            skipped_missing_files.append(image_name + ".xml")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size_node = root.find("size")
        if size_node is not None:
            width = int(float(size_node.find("width").text))
            height = int(float(size_node.find("height").text))
        else:
            width, height = get_image_size(image_path)

        coco["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()

            if cls_name not in class_to_id:
                skipped_unknown_classes.add(cls_name)
                continue

            difficult_node = obj.find("difficult")
            difficult = int(difficult_node.text) if difficult_node is not None else 0

            if ignore_difficult and difficult == 1:
                continue

            bbox = obj.find("bndbox")

            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # VOC bbox: xmin, ymin, xmax, ymax
            # COCO bbox: x, y, width, height
            x = max(0, xmin)
            y = max(0, ymin)
            w = max(0, xmax - xmin)
            h = max(0, ymax - ymin)

            if w <= 0 or h <= 0:
                continue

            area = w * h

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_to_id[cls_name],
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0,
                "segmentation": []
            })

            ann_id += 1

        img_id += 1

    os.makedirs(os.path.dirname(save_json), exist_ok=True)

    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print("VOC to COCO conversion finished.")
    print(f"Split: {split}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])}")
    print(f"Saved to: {save_json}")

    if skipped_unknown_classes:
        print("\nWarning: skipped unknown classes:")
        for cls_name in sorted(skipped_unknown_classes):
            print(f"  - {cls_name}")

    if skipped_missing_files:
        print("\nWarning: skipped missing files:")
        for name in skipped_missing_files[:20]:
            print(f"  - {name}")
        if len(skipped_missing_files) > 20:
            print(f"  ... and {len(skipped_missing_files) - 20} more")


if __name__ == "__main__":
    args = parse_args()

    voc_xml_to_coco(
        voc_root=args.voc_root,
        split=args.split,
        save_json=args.save_json,
        classes=args.classes,
        ignore_difficult=args.ignore_difficult
    )
