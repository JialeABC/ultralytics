from ultralytics import YOLO


if __name__ == '__main__':
    # model = YOLO("D:/Deeplearning_code/yolov8/ultralytics/ultralytics/cfg/models/v8/custom/yolov8_backbone.yaml")  # 从头开始构建新模型
    model = YOLO("runs/detect/train/weights/last.pt")
    results = model.train(data="D:/Deeplearning_code/yolov8/ultralytics/ultralytics/cfg/datasets/coco.yaml", epochs=100,
                          imgsz=1600,resume=True,patience=10)


##使用sobel算子
# import cv2
# import os
# import numpy as np
#
#
# # 创建输出文件夹
# def create_output_folder(output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#
# # 读取图像并转换为灰度图像
# def read_and_convert_to_gray(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"无法读取图像: {image_path}")
#         return None
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return gray_image
#
#
# # 使用Sobel算子计算图像的梯度
# def apply_sobel_operator(image):
#     # 计算图像在x方向和y方向的梯度
#     grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # X方向梯度
#     grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Y方向梯度
#
#     # 计算梯度幅值（即边缘信息）
#     gradient_magnitude = cv2.magnitude(grad_x, grad_y)
#
#     # 转回uint8类型（因为图像的像素值通常是0到255之间的）
#     gradient_magnitude = np.uint8(np.absolute(gradient_magnitude))
#
#     return gradient_magnitude
#
#
# # 处理并保存图像
# def process_and_save_image(image_path, output_folder):
#     image_name = os.path.basename(image_path)  # 获取文件名
#     output_image_path = os.path.join(output_folder, image_name)  # 输出图像路径
#
#     gray_image = read_and_convert_to_gray(image_path)
#     if gray_image is not None:
#         gradient_image = apply_sobel_operator(gray_image)  # 应用Sobel算子
#         cv2.imwrite(output_image_path, gradient_image)  # 保存处理后的图像
#         print(f"处理并保存图像: {output_image_path}")
#     else:
#         print(f"跳过图像: {image_path}")
#
#
# # 主函数，批量处理文件夹中的图像
# def process_images_in_folder(input_folder, output_folder):
#     # 创建输出文件夹
#     create_output_folder(output_folder)
#
#     # 获取输入文件夹中的所有图像文件
#     for filename in os.listdir(input_folder):
#         file_path = os.path.join(input_folder, filename)
#
#         # 检查文件是否为图像文件（可以根据需要扩展文件类型）
#         if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
#             process_and_save_image(file_path, output_folder)
#
#
# ###gamma矫正红外图像
# def inf_enhanced(input_folder, output_folder,gradTh, gamma):
#     create_output_folder(output_folder)
#
#
#     for filename in os.listdir(input_folder):
#         file_path = os.path.join(input_folder, filename)
#         image_name = os.path.basename(file_path)  # 获取文件名
#         output_image_path = os.path.join(output_folder, image_name)  # 输出图像路径
#         gray_image,image = read_and_convert_to_gray(file_path)
#         if gray_image is not None:
#             gradient_image,image = apply_sobel_operator(gray_image)  # 应用Sobel算子
#             height, width = gradient_image.shape
#             hist = np.zeros(256, dtype=int)
#             gradient_image_normalized = gradient_image / np.max(gradient_image)
#             for row in range(height):
#                 for col in range(width):
#                     if gradient_image_normalized[row, col] < gradTh:
#                         gradient_image[row, col] = 0
#                     obj = gradient_image[row, col]
#                     hist[obj] += 1
#
#             hist_normalized = hist / np.sum(hist)
#             hisGrad = np.zeros(256, dtype=float)
#
#             for i in range(len(hist_normalized)):
#                 mask = gradient_image==i
#                 num = np.sum(mask)
#                 hisGrad[i] = num * i * hist_normalized[i]
#             histcum = np.cumsum(hisGrad)
#             histcum_norm = histcum / np.max(histcum)
#             map = histcum_norm
#             map_cor = (map**(1/gamma)+(1-(1-map)**(1/gamma)))/2
#
#             for row in range(height):
#                 for col in range(width):
#                     temp = image[row, col]
#                     image[row, col] = image[row, col] * map_cor[temp]
#             cv2.imwrite(output_image_path, image)
#
#
#
#
#
#
# # 设置输入文件夹和输出文件夹路径
# input_folder = 'D:/Deeplearning_code/ultralytics-main/dataset/test'  # 需要处理的图像文件夹
# output_folder = 'D:/Deeplearning_code/ultralytics-main/dataset/sobel_enhanced'  # 存放梯度图像的文件夹
#
# # 调用主函数处理文件夹中的图像
# process_images_in_folder(input_folder, output_folder)
# import os
# vis_folder="D:/Deeplearning_code/ultralytics-main/dataset/vis_sobel/images"
# grad_folder="D:/Deeplearning_code/ultralytics-main/dataset/vis/images"
#
# for i in os.listdir(grad_folder):
#     path=os.path.join(vis_folder,i)
#     if os.path.exists(path):
#         pass
#     else:
#         print(i)