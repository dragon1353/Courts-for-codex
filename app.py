import html
import os
import subprocess
from datetime import date
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
from rag_agent import query_rag_system
import sys
import threading

# 匯入桌面應用程式視窗模組
import webview
# 匯入原生對話框模組
import tkinter as tk
from tkinter import filedialog

import config

# --- Flask 後端伺服器部分 ---
app = Flask(__name__)

# --- 配置與全域變數 ---
DOWNLOADER_SCRIPT = os.path.join(os.path.dirname(__file__), 'googledata.py')
AUTO_LABEL_SCRIPT = os.path.join(os.path.dirname(__file__), 'local_auto_label.py')
DATA_PROCESSOR_SCRIPT = os.path.join(os.path.dirname(__file__), 'legal_data_processor.py')
TRAIN_MODEL_SCRIPT = os.path.join(os.path.dirname(__file__), 'train_model.py')
VECTORDB_SCRIPT = os.path.join(os.path.dirname(__file__), 'build_vectordb.py')
TEMP_DIR = config.TEMP_DIR
FINAL_REPORT_FILE = config.FINAL_REPORT_FILE

# --- 狀態管理字典 ---
training_status = {"state": "idle", "message": "閒置"}
analysis_status = {"state": "idle", "message": "閒置"}
download_status = {"state": "idle", "message": "閒置"}
status_lock = threading.Lock()


def task_is_running():
    """集中判斷背景任務狀態，避免不同 API 同時啟動高負載工作。"""
    return (
        training_status["state"] in {"queued", "loading"}
        or analysis_status["state"] == "analyzing"
        or download_status["state"] in {"queued", "downloading"}
    )


def run_python_stage(script_path, args, log_prefix, on_output=None):
    """以目前 Python 直譯器執行子程序，串流輸出並回傳結束碼。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        [sys.executable, script_path, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=env,
        bufsize=1,
    )
    if process.stdout is not None:
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            if on_output:
                on_output(line)
            print(f"[{log_prefix}]: {line}")
    process.wait()
    return process.returncode


def parse_roc_date(year, month, day):
    """驗證民國日期並轉成可比較的西元 date。"""
    roc_year = int(year)
    if roc_year < 1:
        raise ValueError("民國年份必須大於 0。")
    return date(roc_year + 1911, int(month), int(day))


def save_final_report(chunks):
    """以暫存檔原子更新報告，避免中途留下半份檔案。"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_report = f"{FINAL_REPORT_FILE}.tmp"
    with open(temp_report, "w", encoding="utf-8") as report_file:
        report_file.write("".join(chunks))
    os.replace(temp_report, FINAL_REPORT_FILE)

# --- 背景任務 ---

def run_download_task(start_y, start_m, start_d, end_y, end_m, end_d):
    """背景任務：非同步執行 googledata.py 並更新 download_status"""
    global download_status
    download_status = {"state": "downloading", "message": "正在初始化下載任務..."}

    try:
        return_code = run_python_stage(
            DOWNLOADER_SCRIPT,
            [start_y, start_m, start_d, end_y, end_m, end_d],
            "googledata",
            lambda line: download_status.update(message=line),
        )
        if return_code == 0:
            download_status["state"] = "success"
            download_status["message"] = "PDF 下載流程執行完畢。"
        else:
            download_status["state"] = "error"
            download_status["message"] = f"下載腳本執行出錯，返回碼: {return_code}"

    except FileNotFoundError:
        download_status["state"] = "error"
        download_status["message"] = f"錯誤：找不到腳本 {DOWNLOADER_SCRIPT}"
    except Exception as e:
        download_status["state"] = "error"
        download_status["message"] = f"啟動下載腳本時發生錯誤: {e}"

# 處理 PDF 標註、統計、模型訓練與向量庫同步作業
def run_training_task(folder_path):
    global training_status
    stages = [
        ("自動標註 PDF", AUTO_LABEL_SCRIPT, [folder_path], "auto_label"),
        ("建立結構化統計資料", DATA_PROCESSOR_SCRIPT, [folder_path], "legal_stats"),
        ("訓練 PyTorch 特徵模型", TRAIN_MODEL_SCRIPT, [], "train_model"),
        ("重建 ChromaDB 向量索引", VECTORDB_SCRIPT, [folder_path], "vectordb"),
    ]

    try:
        for index, (label, script_path, args, log_prefix) in enumerate(stages, start=1):
            stage_prefix = f"階段 {index}/{len(stages)}"
            training_status = {"state": "loading", "message": f"{stage_prefix}：{label}..."}

            def update_progress(line, prefix=stage_prefix):
                training_status["message"] = f"{prefix}：{line}"

            return_code = run_python_stage(script_path, args, log_prefix, update_progress)
            if return_code != 0:
                training_status = {
                    "state": "error",
                    "message": f"{stage_prefix}「{label}」失敗，返回碼 {return_code}。",
                }
                return

        training_status = {
            "state": "success",
            "message": "同步完成：資料集、統計、特徵模型與 ChromaDB 已更新。",
        }

    except Exception as e:
        training_status["state"] = "error"
        training_status["message"] = f"啟動同步任務時發生錯誤: {e}"

#處理 RAG 問答與分析引擎 (串流模式)
def generate_rag_stream(user_prompt):
    """
    Generator 函數，逐步回傳 RAG 分析的 HTML 片段。
    """
    global analysis_status
    report_chunks = []
    try:
        print(f"啟動 RAG 串流分析，提問長度：{len(user_prompt)} 字。")
        for chunk in query_rag_system(user_prompt):
            report_chunks.append(chunk)
            yield chunk
        save_final_report(report_chunks)
        analysis_status = {"state": "success", "message": "分析完成"}
    except GeneratorExit:
        analysis_status = {"state": "error", "message": "分析串流已由用戶端中止。"}
        raise
    except Exception as e:
        analysis_status = {"state": "error", "message": f"分析失敗: {e}"}
        error_chunk = f"<div style='color: red;'>串流中斷: {html.escape(str(e))}</div>"
        report_chunks.append(error_chunk)
        save_final_report(report_chunks)
        yield error_chunk

# 原本的 run_analysis_task 已不再由狀態輪詢驅動，改為直接 Streaming

# --- Flask 路由 ---
@app.route('/')
def index():
    return render_template('index.html')


# --- API 端點 ---
@app.route('/start_download', methods=['POST'])
def start_download_api():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "未提供日期資料。"}), 400

    start_y = data.get('start_year')
    start_m = data.get('start_month')
    start_d = data.get('start_day')
    end_y = data.get('end_year')
    end_m = data.get('end_month')
    end_d = data.get('end_day')

    if not all([start_y, start_m, start_d, end_y, end_m, end_d]):
        return jsonify({"status": "error", "message": "日期參數不完整。"}), 400

    try:
        start_date = parse_roc_date(start_y, start_m, start_d)
        end_date = parse_roc_date(end_y, end_m, end_d)
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "日期格式或日期數值無效。"}), 400
    if start_date > end_date:
        return jsonify({"status": "error", "message": "開始日期不得晚於結束日期。"}), 400

    with status_lock:
        if task_is_running():
            return jsonify({"status": "error", "message": "已有任務在運行中。"}), 409
        download_status.update(state="queued", message="下載任務已排入執行。")

    thread = threading.Thread(
        target=run_download_task,
        args=(start_y, start_m, start_d, end_y, end_m, end_d),
        name="judgment-download",
    )
    thread.daemon = True
    thread.start()
    return jsonify({"status": "success"})

@app.route('/download_status')
def get_download_status_api():
    return jsonify(download_status)


@app.route('/start_training', methods=['POST'])
def start_training_api():
    data = request.get_json(silent=True) or {}
    folder_path = data.get("folder_path")
    if not folder_path:
        return jsonify({"status": "error", "message": "未提供資料夾路徑。"}), 400
    if not os.path.isdir(folder_path):
        return jsonify({"status": "error", "message": "指定的資料夾不存在。"}), 400

    with status_lock:
        if task_is_running():
            return jsonify({"status": "error", "message": "已有任務在運行中。"}), 409
        training_status.update(state="queued", message="同步任務已排入執行。")

    thread = threading.Thread(
        target=run_training_task,
        args=(folder_path,),
        name="legal-data-sync",
    )
    thread.daemon = True
    thread.start()
    return jsonify({"status": "success"})

@app.route('/training_status')
def get_training_status_api():
    return jsonify(training_status)

@app.route('/start_analysis', methods=['POST'])
def start_analysis_api():
    """
    不使用背景 Thread，改用 Flask Response 串流直接回傳內容。
    """
    global analysis_status
    data = request.get_json(silent=True) or {}
    user_prompt = data.get("user_prompt")
    if not user_prompt:
        return jsonify({"status": "error", "message": "請輸入提問或分析指令。"}), 400

    with status_lock:
        if task_is_running():
            return jsonify({"status": "error", "message": "已有任務在運行中。"}), 409
        analysis_status = {"state": "analyzing", "message": "正在分析..."}

    return Response(generate_rag_stream(user_prompt), content_type="text/html; charset=utf-8")

@app.route('/analysis_status')
def get_analysis_status_api():
    return jsonify(analysis_status)

@app.route('/get_final_report')
def get_final_report_api():
    return send_from_directory(TEMP_DIR, os.path.basename(FINAL_REPORT_FILE))


# --- pywebview 桌面應用程式啟動器 ---
class Api:
    def select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="請選擇包含 PDF 的資料夾")
        root.destroy()
        return folder

def run_flask():
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT)

if __name__ == '__main__':
    os.makedirs(TEMP_DIR, exist_ok=True)
    api = Api()
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    webview.create_window(
        config.WINDOW_TITLE,
        f'http://{config.FLASK_HOST}:{config.FLASK_PORT}',
        js_api=api,
        width=config.WINDOW_WIDTH,
        height=config.WINDOW_HEIGHT
    )
    webview.start()
