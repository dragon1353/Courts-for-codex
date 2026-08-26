import os
import sys
from pypdf import PdfReader
import time
import config

# 定義暫存檔案的儲存位置
TEMP_DIR = config.TEMP_DIR
OUTPUT_TEMP_FILE = config.COMBINED_TEXT_FILE

def extract_text_from_all_pdfs(pdf_folder_path):
    """
    從指定資料夾中所有 PDF 檔案提取文字，並合併儲存到一個暫存檔。
    提供詳細的進度和錯誤回報。
    """
    if not os.path.isdir(pdf_folder_path):
        # 使用特定前綴來標記最終錯誤訊息
        print(f"FINAL_ERROR:提供的路徑不是一個有效的資料夾: {pdf_folder_path}")
        return None

    os.makedirs(TEMP_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)
    if total_files == 0:
        print(f"FINAL_ERROR:在 {pdf_folder_path} 中找不到任何 PDF 檔案。")
        return None

    print(f"找到 {total_files} 個 PDF 檔案。開始提取文字...")

    full_text_content = ""
    success_count = 0
    fail_count = 0
    failed_files = []

    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        # 每處理一個檔案就打印進度，讓主程式可以捕捉
        print(f"PROGRESS:{i+1}/{total_files}:{pdf_file}")
        time.sleep(0.05) # 短暫休眠，讓 I/O 有時間反應

        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                full_text_content += page.extract_text() or ""
            full_text_content += "\n\n--- 文件分隔 ---\n\n"
            success_count += 1
        except Exception as e:
            # 記錄錯誤而不是中斷
            fail_count += 1
            failed_files.append(pdf_file)
            print(f"  讀取 {pdf_file} 時發生錯誤: {e}")
            continue

    try:
        with open(OUTPUT_TEMP_FILE, 'w', encoding='utf-8') as f:
            f.write(full_text_content)

        # 打印最終的總結報告
        summary_message = f"處理完成！成功讀取 {success_count} 個檔案，失敗 {fail_count} 個。"
        if fail_count > 0:
            summary_message += f" 失敗檔案列表: {', '.join(failed_files)}"

        # 使用特定前綴來標記最終成功訊息
        print(f"FINAL_SUCCESS:{summary_message}")
        return OUTPUT_TEMP_FILE
    except Exception as e:
        print(f"FINAL_ERROR:儲存暫存檔時發生錯誤: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        extract_text_from_all_pdfs(folder_path)
    else:
        print("FINAL_ERROR:請提供一個資料夾路徑作為命令行參數。")