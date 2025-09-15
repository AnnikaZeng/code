import pandas as pd
import glob
import os

# 1. 指定文件夹路径
folder = "F:\Desktop\GSTAR商品数据\summarize"  # ⚠️ 改成你自己的文件夹路径
file_list = glob.glob(os.path.join(folder, "*.xlsx"))

# 2. 在文件夹下新建 merged/ 子目录（如果不存在就创建）
merged_folder = os.path.join(folder, "merged")
os.makedirs(merged_folder, exist_ok=True)

# 3. 遍历文件
for file in file_list:
    # 跳过已经在 merged/ 文件夹里的
    if os.path.dirname(file) == merged_folder:
        continue

    xls = pd.ExcelFile(file)
    # 合并所有 sheet
    df_merged = pd.concat(
        [xls.parse(sheet) for sheet in xls.sheet_names], ignore_index=True
    )

    # 生成新文件名（放到 merged 文件夹中）
    base = os.path.basename(file)  # e.g. A.xlsx
    name, ext = os.path.splitext(base)  # e.g. ("A", ".xlsx")
    new_file = os.path.join(merged_folder, f"{name}_merged{ext}")

    # 保存
    df_merged.to_excel(new_file, index=False)
    print(f"✅ 已生成: {new_file}")

print("🎉 所有文件已处理完成，结果都在 merged/ 文件夹里。")
