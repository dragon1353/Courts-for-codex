# 台灣司法判決地端 AI 分析工具

這是一套以 Python、Flask 與 PyWebView 建立的 Windows 桌面應用程式，可從司法院法學資料檢索系統下載刑事判決 PDF，並在本機完成資料標註、統計整理、特徵模型訓練、向量索引與 RAG 問答。

## 主要功能

- 依民國年日期區間，透過 Selenium 自動查詢並下載司法院判決 PDF。
- 使用規則關鍵字產生有罪、無罪與未知標籤。
- 擷取罪名、刑期、罰金及判決全文，建立結構化統計資料。
- 訓練字元級雙向 LSTM Autoencoder，產生 256 維本地法律特徵。
- 使用 ChromaDB、字元 N-gram TF-IDF、本地特徵重排及 TurboQuant 檢索判決。
- 串流呼叫本機 Ollama 模型產生法律分析報告。
- 一鍵同步標註資料、統計資料、模型及向量索引。

## 系統流程

```text
司法院網站
  → Selenium 下載 PDF
  → 規則標註 legal_dataset.csv
  → 結構化統計 legal_stats.csv
  → PyTorch 模型 best_model.pth
  → ChromaDB 向量索引
  → ChromaDB + TF-IDF + 本地模型重排
  → Ollama 串流回答
```

## 環境需求

- Windows 10／11
- Python 3.12
- Google Chrome
- Ollama，以及 `config.py` 指定的本地模型；目前預設為 `gemma4`
- 訓練可使用 CUDA GPU，沒有 CUDA 時會改用 CPU

首次執行下載功能時，WebDriver Manager 可能需要連網取得與 Chrome 相容的驅動程式。

## 安裝

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull gemma4
```

## 啟動桌面應用程式

```powershell
python app.py
```

應用程式預設在 `127.0.0.1:5000` 啟動 Flask，並由 PyWebView 開啟桌面視窗。

## 同步資料與模型

在介面選擇 PDF 資料夾後，按下「同步資料、訓練模型並重建索引」。系統會依序執行：

1. `local_auto_label.py`：建立 `dataset/legal_dataset.csv`
2. `legal_data_processor.py`：建立 `dataset/legal_stats.csv`
3. `train_model.py`：建立 `dataset/best_model.pth`
4. `build_vectordb.py`：重建 `dataset/chroma_db/`

也可以分別執行：

```powershell
python local_auto_label.py "D:\judicial_pdfs"
python legal_data_processor.py "D:\judicial_pdfs"
python train_model.py
python build_vectordb.py "D:\judicial_pdfs"
```

可調整的路徑、模型名稱、下載限制、訓練參數及 RAG 門檻集中在 `config.py`。

## 驗證

```powershell
python -B -m pytest -q -p no:cacheprovider
python -B -c "from pathlib import Path; files=[p for p in Path('.').rglob('*.py') if not any(part in {'.git','__pycache__','.agents'} for part in p.parts)]; [compile(p.read_text(encoding='utf-8-sig'), str(p), 'exec') for p in files]; print(f'parsed {len(files)} Python files')"
```

`turboquant_pkg/validate.py` 會下載 Hugging Face 模型並使用大量 GPU 記憶體，不屬於一般測試，請只在準備好對應環境時手動執行。

## 資料與安全

- 判決全文、CSV、模型、ChromaDB 與暫存報告均是本機生成資料，不應加入版本控制或上傳外部服務。
- 既有 Git 歷史可能仍包含舊版生成資料；`.gitignore` 只會阻止新的未追蹤檔案，不會自動移除已追蹤內容。
- RAG 回覆是輔助分析，不構成正式法律意見；重要案件仍應由合格法律專業人員核對。
- 自動下載功能應遵守司法院網站使用規範與合理的存取頻率。
