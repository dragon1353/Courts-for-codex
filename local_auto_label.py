import os
import sys
import glob
import csv
import re
from pypdf import PdfReader

# 載入設定檔取得 PDF 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def rule_based_labeling(text):
    """
    純地端的標註系統：透過法律常用關鍵字正規表示式來自動判斷有罪/無罪。
    您可以根據您的法律知識，隨時擴充這些關鍵字。
    """
    # 法律文書主文通常出現在前面，或者判決主文處
    # 1. 有罪的關鍵字 (只要中一個就當作有罪)
    guilty_keywords = config.GUILTY_KEYWORDS

    # 2. 無罪/免刑的關鍵字
    not_guilty_keywords = config.NOT_GUILTY_KEYWORDS

    # 檢查是否有有罪的詞彙
    for keyword in guilty_keywords:
        if re.search(keyword, text):
            return "有罪" # 在機器學習裡我們通常設成 Label 1

    # 如果沒有明顯的有罪詞彙，再檢查無罪詞彙
    for keyword in not_guilty_keywords:
        if re.search(keyword, text):
            return "無罪" # Label 0

    return "未知" # 若都沒比對到，會被訓練腳本過濾掉

def generate_local_dataset(pdf_folder_path=None):
    pdf_folder_path = pdf_folder_path or config.PDF_SAVE_PATH
    pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if not pdf_files:
        print(f"在 {pdf_folder_path} 找不到任何 PDF 檔案。請先使用爬蟲準備資料。")
        return None

    print(f"開始全地端自動分析與標註...找到 {len(pdf_files)} 個 PDF。")

    output_dir = config.DATASET_DIR
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = config.CSV_DATASET_PATH

    labeled_count = {"有罪": 0, "無罪": 0, "未知": 0}

    with open(csv_file_path, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["FileName", "TextContent", "Label"])

        for i, pdf_path in enumerate(pdf_files):
            filename = os.path.basename(pdf_path)
            raw_text = extract_text_from_pdf(pdf_path)

            if not raw_text.strip():
                continue

            clean_text = raw_text.replace("\r", "").replace("\n", " ").strip()

            # 使用我們寫的地端關鍵字判斷系統
            label = rule_based_labeling(clean_text)
            labeled_count[label] += 1

            writer.writerow([filename, clean_text, label])

            # 因為不到 0.1 秒就分析完了，沒用網路，所以也不需要 sleep 暫停
            if (i+1) % 10 == 0:
                print(f"已處理 {i+1} 筆資料...")

    print("\n地端資料集生成完畢！")
    print(f"資料分佈 (Data Distribution): {labeled_count}")
    print("接下來請執行: python train_model.py 來開始神經網路訓練！")
    return csv_file_path

if __name__ == "__main__":
    selected_folder = sys.argv[1] if len(sys.argv) > 1 else config.PDF_SAVE_PATH
    sys.exit(0 if generate_local_dataset(selected_folder) else 1)
