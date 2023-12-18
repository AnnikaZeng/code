import PyPDF2
import pdfplumber
import pandas as pd


# 打开PDF文件并提取文本
def extract_text_from_pdf(pdf_path):
    pdf_file = open(pdf_path, "rb")
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extractText()
    pdf_file.close()
    return text


# 在提取的文本中搜索关键词并提取包含表格的部分
def extract_tables_with_keywords(text, keywords):
    tables = []
    for keyword in keywords:
        keyword_start = text.find(keyword)
        if keyword_start != -1:
            # Find the page where the keyword appears
            page_num = text.count("\n", 0, keyword_start)
            pdf = pdfplumber.open(pdf_path)
            page = pdf.pages[page_num]
            table = page.extract_table()
            if table:
                tables.append(table)
    return tables


# 清理和处理表格数据
def clean_and_process_tables(tables):
    dfs = []
    for table in tables:
        df = pd.DataFrame(table[1:], columns=table[0])
        dfs.append(df)
    return dfs


# 保存数据
def save_tables_to_csv(dataframes):
    for i, df in enumerate(dataframes):
        df.to_csv(f"table_{i}.csv", index=False)


# 主程序
if __name__ == "__main__":
    pdf_path = "G:/Documents/Zotero_references/武汉科技大学2021/刘世琳_2021_基于ESG视角的城镇污水处理PPP项目绩效评价研究.pdf"
    keywords = ["指标", "评价指标", "评价体系"]
    text = extract_text_from_pdf(pdf_path)
    tables = extract_tables_with_keywords(text, keywords)
    dataframes = clean_and_process_tables(tables)
    save_tables_to_csv(dataframes)
