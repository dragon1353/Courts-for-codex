import os

# --- 基礎網路與視窗設定 ---
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
WINDOW_TITLE = '地端法律 AI 分析工具 (RAG + Unsupervised)'
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 900

# --- 司法院查詢與爬蟲設定 ---
JUDICIAL_TARGET_URL = "https://judgment.judicial.gov.tw/FJUD/Default_AD.aspx"
PDF_SAVE_PATH = r"D:\judicial_pdfs"
MAX_DOWNLOADS = 1000
SELENIUM_WAIT_SECONDS = 20
PDF_DOWNLOAD_WAIT_SECONDS = 5
PAGE_CHANGE_WAIT_SECONDS = 3

# --- 內部路徑設定 ---
BASE_DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
COMBINED_TEXT_FILE = os.path.join(TEMP_DIR, 'combined_pdf_text.txt')
FINAL_REPORT_FILE = os.path.join(TEMP_DIR, 'final_report.html')
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_DATASET_PATH = os.path.join(DATASET_DIR, "legal_dataset.csv")
MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "best_model.pth")
CHROMA_DB_DIR = os.path.join(DATASET_DIR, "chroma_db")

# --- 生成式 AI (Ollama) 整合設定 ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "gemma4"
OLLAMA_KEYWORD_TEMPERATURE = 0.3
OLLAMA_KEYWORD_MAX_TOKENS = 30
OLLAMA_KEYWORD_TIMEOUT_SECONDS = 300
OLLAMA_GENERATION_TIMEOUT_SECONDS = 1200

# --- PyTorch 模型訓練參數 (非監督式特徵學習) ---
TRAIN_MAX_VOCAB_SIZE = 10000
TRAIN_MAX_SEQ_LEN = 1500  # 建議不超過 1份2048字 以免長序列導致顯存溢出
TRAIN_BATCH_SIZE = 128  # 經測試，16GB 顯存可穩定支援到 1次128份數，推薦以此值平衡效能與穩定性
TRAIN_EPOCHS = 50
TRAIN_LEARNING_RATE = 1e-3

# --- 自動標註關鍵字 (用於 local_auto_label.py) ---
GUILTY_KEYWORDS = [
    r"處有期徒刑", r"處拘役", r"處罰金", r"犯.*?罪", r"應執行", r"處有期", r"處刑"
]
NOT_GUILTY_KEYWORDS = [
    r"無罪", r"不受理", r"免訴", r"免刑"
]

# --- RAG 與語意檢索參數 ---
RAG_SIMILARITY_THRESHOLD = 0.05
RAG_MAX_CONTEXT_CHARS = 100000
RAG_MAX_DOC_COUNT = 50
RAG_TFIDF_MAX_FEATURES = 40000
RAG_TFIDF_NGRAM_RANGE = (1, 8)
RAG_TFIDF_CACHE_SIZE = 4
RAG_FEATURE_CACHE_SIZE = 2048
RAG_STREAM_PADDING_BYTES = 16384
VECTOR_CHUNK_SIZE = 800
VECTOR_CHUNK_OVERLAP = 150

# 法律通用罪名關鍵字 (用於精準過濾)
LEGAL_CRIME_WORDS = [
    "殺人", "竊盜", "詐欺", "性剝削", "毒品", "傷害",
    "侵佔", "侵占", "槍砲", "洗錢", "強盜", "偽造", "妨害性隱私"
]
