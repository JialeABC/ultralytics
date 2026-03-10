import os


def strict_convert_labels(input_folder, output_folder, target_class=1):
    # 创建输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ 已创建输出文件夹: {output_folder}")

    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.txt')]

    if not files:
        print("❌ 未找到任何 .txt 文件，请检查路径。")
        return

    print(f"🚀 开始严格处理 {len(files)} 个文件...")

    error_count = 0

    for filename in files:
        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(output_folder, filename)

        try:
            with open(src_path, 'r', encoding='utf-8') as f_in, \
                    open(dst_path, 'w', encoding='utf-8') as f_out:

                line_num = 0
                for line in f_in:
                    line_num += 1
                    # 1. 去除首尾空白符（包括换行符、空格、制表符）
                    clean_line = line.strip()

                    if not clean_line:
                        continue  # 跳过空行

                    # 2. 按空白字符分割（自动处理多个空格或Tab）
                    parts = clean_line.split()

                    if len(parts) < 5:
                        print(f"⚠️  警告: {filename} 第 {line_num} 行格式不正确 (列数<5)，原样保留: {clean_line}")
                        f_out.write(clean_line + '\n')
                        continue

                    # 3. 【核心操作】强制将第一个元素替换为目标类别
                    original_cls = parts[0]
                    parts[0] = str(target_class)

                    # 4. 二次验证（确保真的改成功了）
                    if parts[0] != str(target_class):
                        raise ValueError(f"修改失败！原值:{original_cls}, 目标:{target_class}")

                    # 5. 重组并写入
                    new_line = " ".join(parts)
                    f_out.write(new_line + '\n')

            print(f"✅ 处理完成: {filename}")

        except Exception as e:
            error_count += 1
            print(f"❌ 文件 {filename} 处理出错: {e}")

    print("-" * 30)
    if error_count == 0:
        print(f"🎉 全部成功！所有文件的类别已强制统一为 [{target_class}]")
    else:
        print(f"⚠️  处理结束，但有 {error_count} 个文件出现异常，请检查上方日志。")
    print(f"📂 结果保存在: {output_folder}")


# ================= 配置区域 =================
# 👇 请在这里填入你的真实路径
input_dir = "D:/A_my_study/visdrone/yolov5/labels"  # 👈 修改这里：原始 txt 文件夹路径
output_dir = "D:/A_my_study/visdrone/yolov5/no_class"  # 👈 修改这里：新 txt 保存文件夹路径

# ================= 执行 =================
if __name__ == "__main__":
    if not os.path.exists(input_dir):
        print(f"❌ 错误：找不到输入文件夹 '{input_dir}'")
        print("请检查代码中的 input_dir 路径是否正确。")
    else:
        # 执行转换，目标类别设为 1
        strict_convert_labels(input_dir, output_dir, target_class=0)

        # 💡 额外验证：随机读取一个新文件展示前5行，让你亲眼确认
        import random

        if os.path.exists(output_dir):
            new_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
            if new_files:
                sample_file = random.choice(new_files)
                print(f"\n🔍 随机抽检文件 [{sample_file}] 的前 5 行内容：")
                with open(os.path.join(output_dir, sample_file), 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 5: break
                        print(line.strip())