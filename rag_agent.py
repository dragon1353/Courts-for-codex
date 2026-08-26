import torch
import pandas as pd
import numpy as np
import os
import sys
import json
import html
import hashlib
import time
import threading
import urllib.request
import requests
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- [Google TurboQuant 整合] ---
# 已將庫遷移至專案目錄下的 turboquant_pkg 以解決相對路徑導入問題
try:
    from turboquant_pkg.turboquant import TurboQuantProd
    # 初始化 256 維 (Autoencoder 輸出維度), 4-bit 量化引擎
    tq_engine = TurboQuantProd(d=256, bits=4, device="cpu")
    print("[TurboQuant] 4-bit 量化引擎載入成功")
except Exception as e:
    tq_engine = None
    print(f"[TurboQuant] 載入失敗: {e}")
from models import Vocab, LegalAutoencoder

# 確保載入 config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# 標準化函式與法律用語同義詞對照 (Global Utility)
def normalize_legal(text):
    if not text or not isinstance(text, str): return ""
    # 1. 基礎簡繁與異體字轉換
    t = text.replace("佔", "占").replace("臺", "台").replace("妳", "你")

    # 2. 法律術語同義詞對照 (Synonym Mapping)
    # 將常用口語轉換為法律正式用語，確保 TF-IDF 與硬性過濾能精準對接
    synonyms = {
        "詐騙": "詐欺",
        "騙子": "詐欺",
        "殺": "殺人",
        "偷": "竊盜",
        "拿": "侵占",
        "侵佔": "侵占",
        "性侵": "強制性交",
        "強姦": "強制性交",
        "毒品": "毒品",
        "車禍": "過失傷害", "撞到": "交通", "賠償": "侵權行為"
    }
    for k, v in synonyms.items():
        if k in t:
            t = t.replace(k, v)
    return t.strip().lower()


def javascript_string(value):
    """產生可安全放進內嵌 script 的 JavaScript 字串常值。"""
    return (
        json.dumps(str(value), ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )

# 全域變數載入模型
_model = None
_vocab = None
_model_signature = None
_model_features_cache = OrderedDict()
_csv_cache = {}
_tfidf_index_cache = OrderedDict()
_model_load_lock = threading.RLock()
_feature_cache_lock = threading.RLock()
_retrieval_cache_lock = threading.RLock()
# [優化] 強制將特徵提取模型放在 CPU，避免與 Ollama 搶奪 GPU 顯存導致分析卡死 (死鎖)
_device = torch.device("cpu")


def _file_signature(file_path):
    """回傳足以判斷本機檔案是否更新的輕量簽章。"""
    stat = os.stat(file_path)
    return (os.path.abspath(file_path), stat.st_mtime_ns, stat.st_size)


def _get_csv_snapshot(file_path):
    """依檔案簽章快取 CSV，並在檔案更新後自動失效。"""
    absolute_path = os.path.abspath(file_path)
    signature = _file_signature(absolute_path)
    with _retrieval_cache_lock:
        cached = _csv_cache.get(absolute_path)
        if cached and cached["signature"] == signature:
            return cached

    dataframe = pd.read_csv(absolute_path)
    final_signature = _file_signature(absolute_path)
    if final_signature != signature:
        dataframe = pd.read_csv(absolute_path)
        final_signature = _file_signature(absolute_path)

    snapshot = {
        "path": absolute_path,
        "signature": final_signature,
        "dataframe": dataframe,
    }
    with _retrieval_cache_lock:
        _csv_cache[absolute_path] = snapshot
        stale_keys = [
            key
            for key, value in _tfidf_index_cache.items()
            if value["path"] == absolute_path
            and value["file_signature"] != final_signature
        ]
        for key in stale_keys:
            _tfidf_index_cache.pop(key, None)
    return snapshot


def _load_csv_cached(file_path):
    return _get_csv_snapshot(file_path)["dataframe"]


def _tokenize_chinese(text):
    return " ".join(list(str(text)))


def _get_tfidf_index(file_path, candidate_positions):
    """建立或重用指定候選集合的 TF-IDF 索引。"""
    snapshot = _get_csv_snapshot(file_path)
    dataframe = snapshot["dataframe"]
    required_columns = {"FileName", "TextContent"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"檢索資料缺少欄位：{', '.join(sorted(missing_columns))}")

    positions = np.asarray(candidate_positions, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError("TF-IDF 候選索引必須是一維陣列。")
    if positions.size and (positions.min() < 0 or positions.max() >= len(dataframe)):
        raise IndexError("TF-IDF 候選索引超出資料範圍。")

    positions_hash = hashlib.sha256(positions.tobytes()).digest()
    cache_key = (
        snapshot["signature"],
        tuple(config.RAG_TFIDF_NGRAM_RANGE),
        config.RAG_TFIDF_MAX_FEATURES,
        len(positions),
        positions_hash,
    )
    with _retrieval_cache_lock:
        cached = _tfidf_index_cache.get(cache_key)
        if cached is not None:
            _tfidf_index_cache.move_to_end(cache_key)
            return cached

    candidate_frame = dataframe.iloc[positions]
    texts = candidate_frame["TextContent"].fillna("").astype(str).tolist()
    filenames = candidate_frame["FileName"].fillna("未知來源").astype(str).tolist()
    crime_types = [
        str(candidate_frame.iloc[index].get("CrimeType", "未知"))
        for index in range(len(candidate_frame))
    ]
    tokenized_texts = [_tokenize_chinese(text) for text in texts]
    safe_max_df = 0.9 if len(tokenized_texts) > 1 else 1.0
    vectorizer = TfidfVectorizer(
        max_features=config.RAG_TFIDF_MAX_FEATURES,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=config.RAG_TFIDF_NGRAM_RANGE,
        min_df=1,
        max_df=safe_max_df,
    )
    matrix = vectorizer.fit_transform(tokenized_texts)
    index = {
        "path": snapshot["path"],
        "file_signature": snapshot["signature"],
        "positions": positions,
        "texts": texts,
        "filenames": filenames,
        "crime_types": crime_types,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }

    cache_size = max(0, int(getattr(config, "RAG_TFIDF_CACHE_SIZE", 0)))
    if cache_size:
        with _retrieval_cache_lock:
            _tfidf_index_cache[cache_key] = index
            _tfidf_index_cache.move_to_end(cache_key)
            while len(_tfidf_index_cache) > cache_size:
                _tfidf_index_cache.popitem(last=False)
    return index


def clear_runtime_caches():
    """清除可重建的執行期快取，供資料同步或測試使用。"""
    with _retrieval_cache_lock:
        _csv_cache.clear()
        _tfidf_index_cache.clear()
    with _feature_cache_lock:
        _model_features_cache.clear()


def load_feature_model(force=False):
    global _model, _vocab, _model_signature
    model_path = config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        with _model_load_lock:
            _model = None
            _vocab = None
            _model_signature = None
        with _feature_cache_lock:
            _model_features_cache.clear()
        return False

    signature = _file_signature(model_path)
    with _model_load_lock:
        if not force and _model is not None and _model_signature == signature:
            return True
        try:
            checkpoint = torch.load(model_path, map_location=_device)
            vocab = Vocab(checkpoint['vocab'], checkpoint['inv_vocab'])
            model = LegalAutoencoder(len(vocab.vocab))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(_device)
            model.eval()
            _vocab = vocab
            _model = model
            _model_signature = signature
            with _feature_cache_lock:
                _model_features_cache.clear()
            print(f"成功載入【完全地端】法律特徵模型 (Device: {_device})。")
            return True
        except Exception as e:
            print(f"載入特徵模型失敗: {e}")
            return False

def extract_legal_search_terms(query):
    """
    [智囊模式]: 使用 Llama 3.1 8B 解析使用者的口語問題，並轉換為專業法律檢索詞。
    """
    # 建立精簡的 Prompt 請求 Llama 產出關鍵字
    prompt = f"請將以下口語化的法律問題，轉化為 3-5 個專業的法律檢索關鍵字（例如：案由、法條、行為關鍵字）。只需回覆關鍵字並以空白分隔，其餘廢話都不要。問題：{query}"

    payload = {
        "model": config.OLLAMA_MODEL_NAME, # 修正變數名
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.OLLAMA_KEYWORD_TEMPERATURE,
            "num_predict": config.OLLAMA_KEYWORD_MAX_TOKENS,
        },
    }

    try:
        response = requests.post(
            config.OLLAMA_API_URL,
            json=payload,
            timeout=config.OLLAMA_KEYWORD_TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            # 清理可能的多餘符號
            cleaned_result = result.replace("、", " ").replace(",", " ").replace("：", "")
            print(f"[AI 智慧解析關鍵字]: {cleaned_result}")
            return cleaned_result
    except Exception as e:
        print(f"AI 關鍵字解析失敗: {e}")

    return query # 若失敗則退而求其次使用原句

def get_latent_features(text):
    if (_model is None or _vocab is None) and not load_feature_model():
        return None

    text_value = str(text)
    cache_key = (
        _model_signature,
        hashlib.sha256(text_value.encode("utf-8")).digest(),
    )
    with _feature_cache_lock:
        cached = _model_features_cache.get(cache_key)
        if cached is not None:
            _model_features_cache.move_to_end(cache_key)
            return cached.copy()

    try:
        indices = _vocab.encode(text_value, config.TRAIN_MAX_SEQ_LEN)
        input_tensor = torch.tensor([indices], dtype=torch.long).to(_device)
        with torch.no_grad():
            latent = _model.encode(input_tensor)
        features = latent.cpu().numpy()[0]

        cache_size = max(0, int(getattr(config, "RAG_FEATURE_CACHE_SIZE", 0)))
        if cache_size:
            with _feature_cache_lock:
                _model_features_cache[cache_key] = features
                _model_features_cache.move_to_end(cache_key)
                while len(_model_features_cache) > cache_size:
                    _model_features_cache.popitem(last=False)
        return features.copy()
    except Exception:
        return None

# 初始化
load_feature_model()

class LocalLegalEmbedding(torch.nn.Module):
    # 這裡為了符合 langchain 介面，簡單封裝
    def embed_query(self, text):
        features = get_latent_features(text)
        if features is None:
            raise RuntimeError("本地法律特徵模型尚未載入。")
        return features.tolist()
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

def perform_statistical_analysis(query):
    """
    從 legal_stats.csv 提取統計資訊，提供全局數據。
    【動態模式】：自動感應資料庫中所有罪名，不寫死。
    """
    stats_path = os.path.join(config.DATASET_DIR, "legal_stats.csv")
    if not os.path.exists(stats_path):
        return None

    try:
        df = _load_csv_cached(stats_path)
        total_count = len(df)

        # 1. 動態取得資料庫現有罪名清單 (排除未知)
        unique_crimes = [
            str(crime).strip()
            for crime in df["CrimeType"].unique()
            if pd.notna(crime) and str(crime).strip() not in {"", "未知"}
        ]

        # 2. 定義統計觸發詞 (這部分為語意基礎，較固定)
        quantity_keywords = ["多少", "數量", "份數", "總數", "總共", "幾件", "平均", "比例", "統計", "分析"]
        is_stat_query = any(k in query for k in quantity_keywords)

        # 3. 動態識別使用者想問哪種罪名
        target_crime = None
        for crime in unique_crimes:
            if crime in query:
                target_crime = crime
                break

        # 若兩者都沒對到，則視為非統計題
        if not is_stat_query and not target_crime:
            return None

        return {
            "is_stat_query": is_stat_query,
            "global_total": total_count,
            "target_crime": target_crime,
            "crime_stats": None,
            "top_crimes": df['CrimeType'].value_counts().head(5).to_dict(),
            "global_avg_sentence": round(df['SentenceMonths'].mean(), 1)
        } if (is_stat_query or target_crime) else {
            "is_stat_query": False,
            "global_total": total_count,
            "target_crime": None,
            "crime_stats": None,
            "top_crimes": {},
            "global_avg_sentence": 0
        }
    except Exception as e:
        print(f"統計分析執行失敗: {e}")
        return None

def query_rag_system(user_prompt):
    """
    接收使用者的全局問題，去 ChromaDB 檢索相關判例。
    綜合所有資訊以串流方式丟給 Ollama 生成回答。
    """
    db_dir = config.CHROMA_DB_DIR
    # 訓練由子程序執行；每次新查詢先偵測模型檔案是否已更新。
    load_feature_model()

    # 預設狀態
    source_documents_html = "<p style='color: #666;'>找不到相關知識庫判例。請先執行資料下載與訓練。</p>"
    rag_context = ""

    # === 步驟 0: 結構化統計分析 ===
    analysis_data = perform_statistical_analysis(user_prompt)
    stats_html = ""
    stats_text_for_ai = "（本次問題未觸發特定統計模式）"

    if analysis_data and analysis_data.get("global_total", 0) > 0:
        total = analysis_data["global_total"]
        top_crimes_dict = analysis_data.get("top_crimes", {})
        top_crimes_str = "、".join([f"{k}({v}件)" for k, v in top_crimes_dict.items()])

        # 建立 AI 用的純文字 (作為背景知識，不論是否顯示 UI 都給 AI)
        stats_text_for_ai = f"1. 目前資料庫累積判決總數：{total} 件\n"
        if top_crimes_str:
            stats_text_for_ai += f"2. 主要案件類型分佈：{top_crimes_str}\n"

        if analysis_data.get("target_crime"):
            # 只有在特定的統計查詢或識別到明確罪名時，才在 UI 顯示統計框
            if analysis_data.get("is_stat_query") or analysis_data.get("target_crime"):
                safe_target_crime = html.escape(
                    str(analysis_data["target_crime"]), quote=True
                )
                stats_html = '<div style="background-color: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 8px; margin-bottom: 20px; color: #856404;">'
                stats_html += f"📊 <strong>地端數據庫全局快照</strong>：<br>● <strong>已索引判決總數</strong>：{total} 件<br>"

                # 若有目標罪名的詳細統計，也補上
                # (這裡因為簡化，我們只顯示總量與類型，若需更細則可再擴充)
                stats_html += f"● <strong>偵測到相關類別</strong>：{safe_target_crime}<br>"
                stats_html += "</div>"

            stats_text_for_ai += f"3. 關於使用者詢問的類別：{analysis_data['target_crime']}\n"

    # 若非統計問題，則 stats_html 保持空字串，不干擾場景題 UI

    # === 步驟 1: AI 智慧關鍵字擴張 (Agentic Search) ===
    # [核心優化] 增加 16KB 的填充，強迫跨越所有網路代理層與瀏覽器緩衝區 (確保即時渲染，請勿縮減此處)
    padding = " " * config.RAG_STREAM_PADDING_BYTES
    yield f'''
    <style>
        @keyframes pulse {{ 0% {{ opacity: 0.5; }} 50% {{ opacity: 1; }} 100% {{ opacity: 0.5; }} }}
        .pulse {{ animation: pulse 1.5s infinite; }}
        .step-done {{ color: #28a745; font-weight: bold; margin-top:5px; }}
        .step-active {{ color: #007bff; font-weight: bold; margin-top:5px; }}
        .step-pending {{ color: #999; margin-top:5px; }}
    </style>
    <div id="progress-indicator" style="padding:20px; background-color:#f8f9fa; border-radius:12px; margin-bottom:25px; border:1px solid #dee2e6; font-family: sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <p style="margin:0 0 15px 0; border-bottom:2px solid #eee; padding-bottom:10px; font-size:1.1em;">📋 <b>地端 RAG 執行軌跡 (AI 優先模式)</b></p>
        <div id="step-1" class="step-active pulse">● 階段 1: 正在精準解析提問意圖與法律關鍵字...</div>
        <div id="step-2" class="step-pending">○ 階段 2: 正在生成全量深度文字矩陣 (N-gram 1-8)...</div>
        <div id="step-3" class="step-pending">○ 階段 3: 正在進行全庫相似度掃描與候選召回...</div>
        <div id="step-4" class="step-pending">○ 階段 4: 正在進行本地模型語意重排 (Reranking)...</div>
        <div id="step-5" class="step-pending">○ 階段 5: 正在進行 Ollama 總結分析與法律建議產出...</div>
    </div>
    ''' + padding

    ai_search_keywords = extract_legal_search_terms(user_prompt)
    search_query = normalize_legal(f"{user_prompt} {ai_search_keywords}")

    # 使用 textContent 更新動態文字，避免模型回傳內容成為可執行 HTML。
    step_one_message = javascript_string(f"✅ 階段 1: 法律解析完成：{ai_search_keywords}")
    yield (
        '<script>(function(){const step1=document.getElementById("step-1");'
        f'step1.textContent={step_one_message};'
        'step1.className="step-done";'
        'document.getElementById("step-2").className="step-active pulse";})();</script>'
    )
    yield padding
    time.sleep(0.5)

    # === 步驟 2: PyTorch 特徵提取 (CPU 模式) ===
    print("[Heartbeat] 正在活化地端向量模型 (CPU模式)...")
    latent_vec = get_latent_features(user_prompt)
    feature_status = "✅ 已提取" if latent_vec is not None else "⚠️ 未載入"

    # === 步驟 3: ChromaDB 語意召回 + TF-IDF 精準文本檢索 ===
    yield '<p style="color: #6c757d;">🔎 正在檢索向量庫與資料庫正文，請稍候...</p>'
    relevant_candidates = []
    relevant_docs = []
    try:
        # ChromaDB 提供段落級語意候選；模型或索引不可用時會自動退回 CSV。
        if os.path.isdir(db_dir) and latent_vec is not None:
            try:
                from langchain_community.vectorstores import Chroma

                vectorstore = Chroma(
                    persist_directory=db_dir,
                    embedding_function=LocalLegalEmbedding(),
                )
                vector_hits = vectorstore.similarity_search_with_score(
                    search_query,
                    k=config.RAG_MAX_DOC_COUNT,
                )
                for document, distance in vector_hits:
                    source = document.metadata.get("source", "ChromaDB")
                    filename = os.path.basename(str(source)) or "ChromaDB"
                    relevant_candidates.append({
                        "fname": filename,
                        "content": str(document.page_content),
                        "type": "向量索引",
                        "tfidf_score": 0.0,
                        "vector_score": 1.0 / (1.0 + max(float(distance), 0.0)),
                    })
                del vectorstore
            except Exception as chroma_error:
                print(f"ChromaDB 檢索失敗，改用 CSV：{chroma_error}")

        stats_csv = os.path.join(config.DATASET_DIR, "legal_stats.csv")
        target_csv = stats_csv if os.path.exists(stats_csv) else config.CSV_DATASET_PATH
        scanned_text_count = 0

        if os.path.exists(target_csv):
            df_rag = _load_csv_cached(target_csv)
            required_columns = {"FileName", "TextContent"}
            missing_columns = required_columns.difference(df_rag.columns)
            if missing_columns:
                raise ValueError(f"檢索資料缺少欄位：{', '.join(sorted(missing_columns))}")

            text_series = df_rag["TextContent"].fillna("").astype(str)
            keywords = [keyword for keyword in ai_search_keywords.split() if len(keyword) > 1]
            if keywords:
                mask = text_series.str.contains(
                    keywords[0], na=False, case=False, regex=False
                )
                for keyword in keywords[1:5]:
                    mask |= text_series.str.contains(
                        keyword, na=False, case=False, regex=False
                    )
                candidate_positions = np.flatnonzero(mask.to_numpy())
                if not candidate_positions.size:
                    candidate_positions = np.arange(len(df_rag), dtype=np.int64)
            else:
                candidate_positions = np.arange(len(df_rag), dtype=np.int64)

            scanned_text_count = len(candidate_positions)
            if scanned_text_count:
                ngram_min, ngram_max = config.RAG_TFIDF_NGRAM_RANGE

                yield (
                    f'<!-- PROGRESS: 正在載入或建立深度文字矩陣 '
                    f'(N-gram {ngram_min}-{ngram_max})... -->'
                )
                tfidf_index = _get_tfidf_index(target_csv, candidate_positions)
                query_tokenized = _tokenize_chinese(ai_search_keywords or user_prompt)
                query_vec = tfidf_index["vectorizer"].transform([query_tokenized])
                cosine_sim = cosine_similarity(
                    query_vec, tfidf_index["matrix"]
                ).flatten()
                passed_indices = np.where(
                    cosine_sim > config.RAG_SIMILARITY_THRESHOLD
                )[0]
                sorted_indices = passed_indices[
                    np.argsort(cosine_sim[passed_indices])[::-1]
                ]

                for idx in sorted_indices:
                    relevant_candidates.append({
                        "fname": tfidf_index["filenames"][idx],
                        "content": tfidf_index["texts"][idx],
                        "type": tfidf_index["crime_types"][idx],
                        "tfidf_score": float(cosine_sim[idx]),
                        "vector_score": 0.0,
                    })

        step_two_message = javascript_string(
            "✅ 階段 2: ChromaDB 與 TF-IDF 候選矩陣已完成"
        )
        yield (
            '<script>(function(){const step2=document.getElementById("step-2");'
            f'step2.textContent={step_two_message};step2.className="step-done";'
            'document.getElementById("step-3").className="step-active pulse";})();</script>'
        )
        yield padding

        # 合併兩種召回結果，並先限制需進行深度重排的候選數量。
        unique_candidates = {}
        for candidate in relevant_candidates:
            key = (candidate["fname"], candidate["content"][:300])
            existing = unique_candidates.get(key)
            retrieval_score = max(
                candidate.get("tfidf_score", 0.0), candidate.get("vector_score", 0.0)
            )
            if existing is None or retrieval_score > existing["retrieval_score"]:
                candidate["retrieval_score"] = retrieval_score
                unique_candidates[key] = candidate

        relevant_candidates = sorted(
            unique_candidates.values(),
            key=lambda item: item["retrieval_score"],
            reverse=True,
        )[: max(config.RAG_MAX_DOC_COUNT * 4, config.RAG_MAX_DOC_COUNT)]

        candidate_count = len(relevant_candidates)
        step_three_message = javascript_string(
            f"✅ 階段 3: 完成 {scanned_text_count} 份正文掃描，合併 {candidate_count} 份候選"
        )
        yield (
            '<script>(function(){const step3=document.getElementById("step-3");'
            f'step3.textContent={step_three_message};step3.className="step-done";'
            'document.getElementById("step-4").className="step-active pulse";})();</script>'
        )
        yield f'<!-- PROGRESS: 找到 {candidate_count} 份潛在資料，準備語意排序... -->'
        yield padding

        q_feat = get_latent_features(user_prompt)
        if relevant_candidates and q_feat is not None:
            total_candidates = len(relevant_candidates)
            q_tensor = torch.from_numpy(np.asarray(q_feat)).float()
            for index, document in enumerate(relevant_candidates, start=1):
                if index % 20 == 0:
                    progress_message = f"正在進行深度語意比對 ({index}/{total_candidates})"
                    yield f'<!-- PROGRESS: {progress_message} -->'
                    safe_progress = javascript_string(f"● 階段 4: {progress_message}...")
                    yield (
                        '<script>document.getElementById("step-4").textContent='
                        f'{safe_progress};</script>'
                    )
                    yield padding

                document_features = get_latent_features(document["content"])
                if document_features is None:
                    document["semantic_score"] = document["retrieval_score"]
                    continue
                if tq_engine:
                    document_tensor = torch.from_numpy(
                        np.asarray(document_features)
                    ).float()
                    compressed_document = tq_engine.quantize(document_tensor)
                    score = float(
                        tq_engine.inner_product(q_tensor, compressed_document)
                    )
                else:
                    score = float(
                        cosine_similarity([q_feat], [document_features])[0][0]
                    )
                document["semantic_score"] = score

            relevant_candidates.sort(
                key=lambda item: item.get("semantic_score", item["retrieval_score"]),
                reverse=True,
            )
            rerank_message = "✅ 階段 4: 語意重排完成，已精選最佳參考判決"
        else:
            rerank_message = "⚠️ 階段 4: 本地模型不可用，已依檢索分數排序"

        safe_rerank_message = javascript_string(rerank_message)
        yield (
            '<script>(function(){const step4=document.getElementById("step-4");'
            f'step4.textContent={safe_rerank_message};step4.className="step-done";'
            'document.getElementById("step-5").className="step-active pulse";})();</script>'
        )
        yield padding

        current_chars = 0
        seen_files = set()
        for document in relevant_candidates:
            if len(relevant_docs) >= config.RAG_MAX_DOC_COUNT:
                break
            remaining_chars = config.RAG_MAX_CONTEXT_CHARS - current_chars
            if remaining_chars <= 0:
                break
            filename = document["fname"]
            if filename in seen_files:
                continue
            content = document["content"][:remaining_chars]
            if not content:
                continue
            selected_document = dict(document)
            selected_document["content"] = content
            relevant_docs.append(selected_document)
            seen_files.add(filename)
            current_chars += len(content)

        if relevant_docs:
            source_documents_html = "<ul>"
            for index, document in enumerate(relevant_docs, start=1):
                rag_context += (
                    f"【參考資料 {index} - {document['fname']} "
                    f"(庫存案由: {document['type']})】：\n{document['content']}\n\n"
                )
                safe_filename = html.escape(str(document["fname"]), quote=True)
                source_documents_html += f"<li><strong>{safe_filename}</strong></li>"
            source_documents_html += "</ul>"
            print(
                f"[RAG] 已選取 {len(relevant_docs)} 份參考資料，"
                f"上下文共 {current_chars} 字。"
            )
        else:
            source_documents_html = (
                "<p style='color: #666;'>目前資料庫中查無達到門檻的相關判例。</p>"
            )

    except Exception as error:
        safe_error = html.escape(str(error), quote=True)
        source_documents_html = f"<span style='color: red;'>檢索異常: {safe_error}</span>"

    # === 步驟 5: 組合反幻覺 Prompt 並交由 Ollama 生成 ===
    try:
        system_prompt = f"""
您是台灣法律諮詢專家。請針對使用者的問題：「{user_prompt}」，結合以下資料提供分析。

【⚠ 重要反幻覺與指令遵循準則】：
1. **指令優先**：使用者問題中若包含「請告知我是參照哪一個檔案」、「請列出具體案號」等指令，請**務必遵守**。
2. **主題相關性過濾 (核心防死鎖)**：請先核對【資料來源 1】的主體內容與使用者的「事實」是否具備實質相關性。
   - **如果完全不對題**（例如問性侵卻給光碟案），請明確告知：「目前地端資料庫查無與此特定情節相關的判例，暫無法基於資料庫內容進行分析。」，**絕對禁止** 強行套用不對題的案例進行法律論證。
3. **事實對位分析**：如果資料相關，請拿使用者描述的事實與【資料來源 1】中的情節進行比對分析。

【資料來源 0：地端資料庫統計背景】：
{stats_text_for_ai}

【資料來源 1：地端 PDF 判例摘要 (優先參考)】：
{rag_context if rag_context else "（目前地端資料庫查無高度相符的案例片斷）"}

【資料來源 2：台灣法律體系發散背景】：
- 請將上述案例與您的法律知識（民事、刑事、家事、性侵防治法等）對齊，給予具體的定罪建議或行動方針。

【回覆準則】：
1. **明確引用**：請務必明確指出您是參照哪一個檔案（如：XXX.pdf）進行的比對分析。
2. **純文字格式**：請直接以文字輸出，**絕對禁止** 使用任何 HTML 標籤（如 <p>, <strong>, <ul> 等）或 Markdown 語法（如 #, *, >）。
3. **分段規範**：段落之間請「空一行」（使用一個換行符號），確保閱讀清晰。
"""

        # 先 Yield 報告的開頭與來源部分
        header_html = f'''
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #2c3e50; text-align: center; border-bottom: 3px solid #34495e; padding-bottom: 10px;">⚖️ 智慧法律深度分析報告 (RAG + Stats)</h2>

            {stats_html}

            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #ddd; padding-bottom: 5px;">
                <h3 style="color: #444; margin: 0;">📚 知識庫檢索來源</h3>
                <span style="font-size: 0.8em; color: #666;">地端特徵提取: {feature_status}</span>
            </div>
            <div style="padding: 10px 20px; border-radius: 8px; background-color: #e9ecef; border-left: 5px solid #6c757d; font-size: 0.9em; margin-bottom: 20px; margin-top: 10px;">
                {source_documents_html}
            </div>
            <h3 style="color: #444; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px;">💡 總結分析與建議 (Ollama: {html.escape(config.OLLAMA_MODEL_NAME, quote=True)})</h3>
            <div id="streaming-content" style="padding: 20px; border-radius: 8px; background-color: #ffffff; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); white-space: pre-wrap;">
        '''
        yield header_html

        # 向 Ollama 發送串流請求
        req_data = json.dumps({
            "model": config.OLLAMA_MODEL_NAME,
            "prompt": system_prompt,
            "stream": True
        }).encode('utf-8')

        req = urllib.request.Request(
            config.OLLAMA_API_URL,
            data=req_data,
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(
            req, timeout=config.OLLAMA_GENERATION_TIMEOUT_SECONDS
        ) as response:
            for line in response:
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    response_text = chunk.get("response", "")
                    if response_text:
                        # --- [修正] 改用針對性濾網，避免移除 HTML 標籤的 > ---
                        cleaned_chunk = response_text.replace(">>", "").replace("###", "").replace("**", "").replace("__", "")
                        # 處理行首引用符號，加個空格判斷比較保險
                        cleaned_chunk = cleaned_chunk.replace("\n>", "\n").replace(" > ", " ")
                        yield html.escape(cleaned_chunk)
                    if chunk.get("done", False):
                        break

        # 標記最後階段完成
        yield '<script>document.getElementById("step-5").textContent = "✅ 階段 5: Ollama 總結分析與建議產出完成"; document.getElementById("step-5").className = "step-done";</script>'
        yield padding

        # 結尾 HTML
        footer_html = f'''
            </div>
            <div style="margin-top: 40px; padding: 10px; background-color: #e2e3e5; border-radius: 5px; text-align: center;">
                <p style="color: #6a6c6f; font-size: 0.85em; margin-bottom: 0;">
                    本次分析由 <b>ChromaDB + TF-IDF + 地端特徵重排 + Ollama</b> 驅動。所有內容皆在本地生成。
                </p>
            </div>
        </div>
        '''
        yield footer_html

    except Exception as e:
        raise RuntimeError(f"Ollama 連線或生成失敗：{e}") from e

if __name__ == "__main__":
    # 測試腳本
    for chunk in query_rag_system("測試問題"):
        print(chunk, end="", flush=True)
