import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import sys # 導入 sys 模組用於獲取命令行參數
import config

# --- 變數設定區 ---
# 司法院進階查詢網址
target_url = config.JUDICIAL_TARGET_URL
# PDF 儲存路徑 (請確保此資料夾已存在)
pdf_save_path = config.PDF_SAVE_PATH

# --- Chrome 選項設定 ---
chrome_options = Options()
# 執行完畢後瀏覽器不會自動關閉
chrome_options.add_experimental_option('detach', True)

# 設定下載路徑與PDF處理方式
prefs = {
    # 設定預設下載路徑
    "download.default_directory": pdf_save_path,
    # 停用 Chrome 的內建 PDF 檢視器，強制直接下載
    "plugins.always_open_pdf_externally": True,
    # 關閉下載前的詢問視窗
    "download.prompt_for_download": False,
    # 停用安全瀏覽的下載頁尾 (有時會干擾自動化)
    "safeBrowse.enabled": True
}
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument("--start-maximized")

# --- 全局變數用於 WebDriver 實例 (避免每次調用 perform_judicial_query 都初始化) ---
_driver = None
_wait = None

def init_webdriver():
    """初始化並返回 WebDriver 實例和 WebDriverWait 實例。"""
    global _driver, _wait
    if _driver is None:
        print("正在自動設定 ChromeDriver...")
        try:
            if not os.path.exists(pdf_save_path):
                os.makedirs(pdf_save_path)
                print(f"已建立資料夾: {pdf_save_path}")

            service = Service(ChromeDriverManager().install())
            _driver = webdriver.Chrome(service=service, options=chrome_options)
            _wait = WebDriverWait(_driver, config.SELENIUM_WAIT_SECONDS)
            print("ChromeDriver 設定完成。")
            return True
        except Exception as e:
            print(f"WebDriver 初始化失敗: {e}")
            _driver = None # 確保失敗時重置
            _wait = None
            return False
    return True # 如果已經初始化，直接返回 True

def close_webdriver():
    """關閉 WebDriver 實例。"""
    global _driver
    if _driver:
        _driver.quit()
        _driver = None
        print("ChromeDriver 已關閉。")


def perform_judicial_query(start_year, start_month, start_day, end_year, end_month, end_day):
    """
    使用 Selenium 自動化執行司法院判決書查詢，並批次下載當前頁面所有結果的PDF。
    接收起訖年月日參數。
    """
    global _driver, _wait

    if not init_webdriver():
        return "WebDriver 初始化失敗，無法執行查詢。"

    driver = _driver
    wait = _wait

    try:
        print(f"正在開啟網址: {target_url}")
        driver.get(target_url)

        criminal_case_checkbox_xpath = "//input[@name='jud_sys' and @value='M']"
        print("等待頁面主要元素載入...")
        wait.until(EC.presence_of_element_located((By.XPATH, criminal_case_checkbox_xpath)))
        print("頁面已載入。")

        print("點擊 '刑事' 案件類別...")
        criminal_checkbox = wait.until(EC.element_to_be_clickable((By.XPATH, criminal_case_checkbox_xpath)))
        criminal_checkbox.click()
        print("- 已成功勾選 '刑事'。")

        # --- 使用傳入的參數填寫起訖年月日 ---
        print(f"輸入裁判期間: 民國 {start_year}.{start_month}.{start_day} 至 {end_year}.{end_month}.{end_day}")

        driver.find_element(By.ID, "dy1").send_keys(start_year)
        driver.find_element(By.ID, "dm1").send_keys(start_month)
        driver.find_element(By.ID, "dd1").send_keys(start_day)
        driver.find_element(By.ID, "dy2").send_keys(end_year)
        driver.find_element(By.ID, "dm2").send_keys(end_month)
        driver.find_element(By.ID, "dd2").send_keys(end_day)
        print("- 已輸入裁判期間。")
        # --- 日期輸入欄位替換結束 ---

        print("點擊 '送出查詢' 按鈕...")
        submit_button = wait.until(EC.element_to_be_clickable((By.ID, "btnQry")))
        driver.execute_script("arguments[0].click();", submit_button)
        print("查詢已送出。")

        print("\n等待並切換到結果 iframe...")
        try:
            wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "iframe-data")))
            print("- 已成功切換到 iframe。")
        except Exception as e:
            print(f"切換到 iframe 時失敗: {e}")
            return "切換到 iframe 時失敗。"

        total_downloaded = 0
        max_downloads = config.MAX_DOWNLOADS
        page_num = 1

        while total_downloaded < max_downloads:
            try:
                result_links_selector = "a.hlTitle_scroll"
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, result_links_selector)))
                results_count = len(driver.find_elements(By.CSS_SELECTOR, result_links_selector))

                if results_count == 0:
                    print("在目前頁面中找不到任何查詢結果。")
                    break

                print(f"--- 偵測到第 {page_num} 頁共有 {results_count} 筆資料 ---")
            except TimeoutException:
                print("在目前頁面中找不到任何查詢結果或載入超時。\n")
                break

            for i in range(results_count):
                if total_downloaded >= max_downloads:
                    break

                print(f"\n--- 正在處理全域第 {total_downloaded + 1} / {max_downloads} 筆資料 ---")

                try:
                    all_links = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, result_links_selector)))
                    link_to_click = all_links[i]
                    result_title = link_to_click.text
                    print(f"- 正在點擊: {result_title}")
                    link_to_click.click()

                    print("- 在詳細頁面中尋找並點擊 '轉存PDF'...")
                    pdf_button = wait.until(EC.element_to_be_clickable((By.ID, "hlExportPDF")))
                    driver.execute_script("arguments[0].click();", pdf_button)
                    print(f"- 已點擊 '轉存PDF'。檔案將會下載至 {pdf_save_path}")
                    time.sleep(config.PDF_DOWNLOAD_WAIT_SECONDS)

                    total_downloaded += 1

                    print("- 返回查詢結果列表頁面...")
                    driver.back()

                    print("- 重新切換回 iframe...")
                    wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "iframe-data")))
                except StaleElementReferenceException:
                    print(f"**警告**：第 {total_downloaded + 1} 筆資料的元素已過時，跳過此筆。\n")
                    continue
                except TimeoutException:
                    print(f"**錯誤**：處理第 {total_downloaded + 1} 筆資料時發生超時，跳過此筆。\n")
                    #driver.back()
                    #wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "iframe-data")))
                    continue
                except Exception as e:
                    print(f"**錯誤**：處理第 {total_downloaded + 1} 筆資料時發生未知錯誤: {e}，跳過此筆。\n")
                    driver.back()
                    wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "iframe-data")))
                    continue

            if total_downloaded < max_downloads:
                try:
                    print("\n- 準備切換至下一頁...")
                    next_btn = driver.find_element(By.ID, "hlNext")
                    driver.execute_script("arguments[0].click();", next_btn)
                    time.sleep(config.PAGE_CHANGE_WAIT_SECONDS)
                    page_num += 1
                except Exception as e:
                    print("- 已經沒有下一頁或無法切換下一頁。")
                    break

        driver.switch_to.default_content()
        print(f"\n批次下載流程完成。已下載 {total_downloaded} 份 PDF。瀏覽器將保持開啟狀態供您檢視。\n")
        return f"下載完成。已下載 {total_downloaded} 份 PDF。"

    except Exception as e:
        print(f"執行過程中發生未預期的錯誤: {e}\n")
        return f"執行過程中發生錯誤: {e}"

# --- 主程式執行區 (用於獨立運行測試) ---
if __name__ == '__main__':
    # 當 googledata.py 作為獨立腳本運行時，可以從命令行參數獲取日期
    import sys # 導入 sys 模組用於獲取命令行參數
    from datetime import date
    if len(sys.argv) == 7: # 檢查是否有足夠的參數 (腳本名 + 6個日期組件)
        start_y, start_m, start_d, end_y, end_m, end_d = sys.argv[1:7]
        perform_judicial_query(start_y, start_m, start_d, end_y, end_m, end_d)
    else:
        print("請提供起訖年月日作為命令行參數。例如：python googledata.py 113 01 01 113 01 31")
        # 預設執行，用於快速測試 (使用今天的日期作為範例)
        today = date.today()
        roc_year = today.year - 1911
        perform_judicial_query(str(roc_year), today.strftime('%m'), '01', str(roc_year), today.strftime('%m'), today.strftime('%d'))

    # 執行完畢後，可以選擇關閉瀏覽器
    # close_webdriver()
