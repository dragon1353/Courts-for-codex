import os
import sys
import glob
import re
from pypdf import PdfReader
import pandas as pd

# 載入設定檔取得路徑
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

def extract_crime_info(text):
    """
    使用正則表達式提取罪名、刑期(月)、罰金。
    這是一個具備法律邏輯的解析器。
    """
    # 預設值
    crime_type = "未知"
    sentence_months = 0
    penalty = 0

    # 1. 提取案由 (通常在開頭：因...案件)
    # 這是比較通用的做法，不論是民事或刑事通常都有案由
    main_text = text[:3000]

    # 案由解析 (因...案件 或 因...而...案件)
    case_reason_match = re.search(r"因(.*?)(案件|案件|罪)", main_text)
    if case_reason_match:
        crime_type = case_reason_match.group(1).split("，")[0].strip() # 避免抓太長
    else:
        # 如果沒抓到案由，再回退到原本的「犯...罪」
        crime_match = re.search(r"犯(.*?罪)", main_text)
        if crime_match:
            crime_type = crime_match.group(1)
        else:
            crime_type = "未知"

    # 二次清理：如果是「如附表...」，我們還是保留，但至少這不是硬編碼
    if len(crime_type) > 30: # 如果太長可能解析錯誤，抓前 20 字
        crime_type = crime_type[:20] + "..."

    # 2. 提取刑期 (有期徒刑...年...月)
    year_match = re.search(r"處有期徒刑(?:(\d+)年)?(?:(\d+)月)?", main_text)
    if year_match:
        years = int(year_match.group(1)) if year_match.group(1) else 0
        months = int(year_match.group(2)) if year_match.group(2) else 0
        sentence_months = years * 12 + months
    else:
        # 處理拘役 (通常以日計，我們換算成月，大致 30 日一月)
        detention_match = re.search(r"處拘役(\d+)日", main_text)
        if detention_match:
            sentence_months = round(int(detention_match.group(1)) / 30.0, 1)

    # 3. 提取罰金 (科罰金新臺幣...元)
    penalty_match = re.search(r"科罰金新臺幣([\d,]+|[一二三四五六七八九十百千萬]+)元", main_text)
    if penalty_match:
        val_str = penalty_match.group(1).replace(",", "")
        # 這裡簡單處理數字，實際上可能需要更多繁體中文數字轉換邏輯
        try:
            penalty = int(val_str)
        except:
            # 如果是中文字，暫標為 -1 代表需進階解析
            penalty = -1

    return crime_type, sentence_months, penalty

def process_all_judgments(pdf_folder_path=None):
    pdf_folder_path = pdf_folder_path or config.PDF_SAVE_PATH
    pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if not pdf_files:
        print(f"在 {pdf_folder_path} 找不到 PDF 檔案。")
        return None

    print(f"正在啟動深度結構化分析... 處理 {len(pdf_files)} 份判決書")

    results = []

    for i, pdf_path in enumerate(pdf_files):
        filename = os.path.basename(pdf_path)
        text = extract_text_from_pdf(pdf_path)

        if not text.strip(): continue

        # 清理文字
        clean_text = text.replace("\r", "").replace("\n", " ").strip()

        # 提取結構化數據
        crime, months, money = extract_crime_info(clean_text)

        results.append({
            "FileName": filename,
            "CrimeType": crime,
            "SentenceMonths": months,
            "Penalty": money,
            "TextContent": clean_text  # 無上限儲存完整 PDF 內文
        })

        if (i+1) % 10 == 0:
            print(f"分析中... {i+1}/{len(pdf_files)}")

    # 存檔
    output_path = os.path.join(config.DATASET_DIR, "legal_stats.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n結構化分析完成！")
    print(f"統計報表已產出：{output_path}")
    print("系統現在具備了針對罪名與刑期的統計與分析能力。")
    return output_path

if __name__ == "__main__":
    selected_folder = sys.argv[1] if len(sys.argv) > 1 else config.PDF_SAVE_PATH
    sys.exit(0 if process_all_judgments(selected_folder) else 1)
