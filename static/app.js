document.addEventListener('DOMContentLoaded', function() {
    // --- 元素選擇 (新增 downloadPdfsBtn) ---
    const selectPathBtn = document.getElementById('selectPathBtn');
    const folderPathInput = document.getElementById('folderPathInput');
    const loadDataBtn = document.getElementById('loadDataBtn');
    const runAnalysisBtn = document.getElementById('runAnalysisBtn');
    const downloadPdfsBtn = document.getElementById('downloadPdfsBtn');
    const dialogInput = document.getElementById('dialogInput');
    const statusBox = document.getElementById('status-box');
    const reportPreview = document.getElementById('reportPreview');

    function escapeHtml(value) {
        return String(value)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    // --- 新增: 日期輸入框元素 ---
    const start_y = document.getElementById('start_year');
    const start_m = document.getElementById('start_month');
    const start_d = document.getElementById('start_day');
    const end_y = document.getElementById('end_year');
    const end_m = document.getElementById('end_month');
    const end_d = document.getElementById('end_day');

    // --- 新增：設定預設日期的邏輯 ---
    function setDefaultDates() {
        const today = new Date();
        const sevenDaysAgo = new Date();
        sevenDaysAgo.setDate(today.getDate() - 7);

        // 轉換為民國年
        const toROCYear = (gregorianYear) => gregorianYear - 1911;

        // 格式化為兩位數字 (例如 01, 02)
        const formatTwoDigits = (num) => num.toString().padStart(2, '0');

        // 設定結束日期為今天
        end_y.value = toROCYear(today.getFullYear());
        end_m.value = formatTwoDigits(today.getMonth() + 1); // 月份從 0 開始
        end_d.value = formatTwoDigits(today.getDate());

        // 設定開始日期為七天前
        start_y.value = toROCYear(sevenDaysAgo.getFullYear());
        start_m.value = formatTwoDigits(sevenDaysAgo.getMonth() + 1);
        start_d.value = formatTwoDigits(sevenDaysAgo.getDate());
    }

    setDefaultDates(); // 在 DOM 內容載入後立即呼叫此函數


    let loadingPoller = null;
    let analysisPoller = null;
    let downloadPoller = null; // 新增 download Poller

    selectPathBtn.addEventListener('click', async function() {
        try {
            const folder = await window.pywebview.api.select_folder();
            if (folder) {
                folderPathInput.value = folder;
            }
        } catch (e) {
            console.error("選擇資料夾時發生錯誤:", e);
            alert("無法呼叫原生 API。請確認您是在 pywebview 視窗中執行。");
        }
    });

    function setButtonsDisabled(disabled) {
        loadDataBtn.disabled = disabled;
        runAnalysisBtn.disabled = disabled;
        selectPathBtn.disabled = disabled;
        downloadPdfsBtn.disabled = disabled; // 同時禁用新按鈕
    }

    // --- 修改: "下載判決書PDF" 按鈕邏輯，以包含日期 ---
    downloadPdfsBtn.addEventListener('click', function() {
        // 獲取日期值
        const dates = {
            start_year: start_y.value.trim(),
            start_month: start_m.value.trim(),
            start_day: start_d.value.trim(),
            end_year: end_y.value.trim(),
            end_month: end_m.value.trim(),
            end_day: end_d.value.trim()
        };

        // 簡單驗證
        for (const key in dates) {
            if (!dates[key]) {
                alert('所有日期欄位都必須填寫！');
                return;
            }
        }

        setButtonsDisabled(true);
        statusBox.textContent = '正在啟動 PDF 下載任務...';
        reportPreview.srcdoc = '';

        // --- 修改 fetch，將日期作為 body 發送 ---
        fetch('/start_download', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(dates)
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                if (downloadPoller) clearInterval(downloadPoller);
                downloadPoller = setInterval(pollDownloadStatus, 1000);
            } else {
                statusBox.textContent = `啟動失敗: ${data.message}`;
                statusBox.style.color = '#dc3545';
                setButtonsDisabled(false);
            }
        })
        .catch(err => {
            statusBox.textContent = '下載請求失敗: ' + err;
            setButtonsDisabled(false);
        });
    });

    function pollDownloadStatus() {
        fetch('/download_status')
        .then(res => res.json())
        .then(data => {
            statusBox.textContent = `[下載進度] ${data.message}`;
            if (data.state === 'success' || data.state === 'error') {
                statusBox.style.color = data.state === 'success' ? '#28a745' : '#dc3545';
                clearInterval(downloadPoller);
                setButtonsDisabled(false);
            } else {
                statusBox.style.color = '#ff9800'; // 橘色
            }
        });
    }

    loadDataBtn.addEventListener('click', function() {
        const folderPath = folderPathInput.value;
        if (!folderPath.trim()) { alert('請輸入或選擇一個資料夾路徑。'); return; }

        setButtonsDisabled(true);
        statusBox.textContent = '正在啟動完整同步任務...';
        reportPreview.srcdoc = ''; // 清空舊報告

        fetch('/start_training', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({folder_path: folderPath})
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                if (loadingPoller) clearInterval(loadingPoller);
                loadingPoller = setInterval(pollTrainingStatus, 1000);
            } else {
                statusBox.textContent = `啟動失敗: ${data.message}`;
                statusBox.style.color = '#dc3545';
                setButtonsDisabled(false);
            }
        })
        .catch(err => {
            statusBox.textContent = '上傳請求失敗: ' + err;
            setButtonsDisabled(false);
        });
    });

    function pollTrainingStatus() {
        fetch('/training_status')
        .then(res => res.json())
        .then(data => {
            statusBox.textContent = `[訓練進度] ${data.message}`;
            if (data.state === 'success' || data.state === 'error') {
                statusBox.style.color = data.state === 'success' ? '#28a745' : '#dc3545';
                clearInterval(loadingPoller);
                setButtonsDisabled(false);
            } else {
                statusBox.style.color = '#007bff';
            }
        });
    }

    runAnalysisBtn.addEventListener('click', async function() {
        const userPrompt = dialogInput.value;
        if (!userPrompt.trim()) { alert('請輸入分析指令才能執行分析。'); return; }

        setButtonsDisabled(true);
        statusBox.textContent = '分析中... 正在載入知識庫與生成報告...';
        statusBox.style.color = '#007bff';

        // 初始化報告區域 (loading 狀態)
        reportPreview.srcdoc = `
            <div style="padding:20px; font-family: sans-serif; text-align:center;">
                <p style="color: #666;">⏳ 正在連線到 Ollama 並搜尋知識庫，請稍候...</p>
            </div>
        `;

        try {
            const response = await fetch('/start_analysis', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({user_prompt: userPrompt})
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.message || '伺服器錯誤');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let isFirstChunk = true;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });

                // 1. 同步更新外部狀態框的文字內容 (Progress Heartbeat)
                const progressMatch = chunk.match(/<!-- PROGRESS: (.*?) -->/);
                if (progressMatch) {
                    statusBox.textContent = `[分析進度] ${progressMatch[1]}`;
                }

                if (isFirstChunk) {
                    // 第一個片段包含完整骨架
                    reportPreview.srcdoc = chunk;
                    isFirstChunk = false;
                } else {
                    const iframeDoc = reportPreview.contentDocument || reportPreview.contentWindow.document;
                    // 目標容器：若階段 1-4 則先貼到 body，階段 5 則貼到 streaming-content
                    let target = iframeDoc.getElementById('streaming-content') || iframeDoc.body;

                    if (target) {
                        // [核心修復] 使用 Range 來解析 HTML，這會強迫瀏覽器執行裡面的 <script>
                        const range = iframeDoc.createRange();
                        range.setStart(target, target.childNodes.length);
                        const fragment = range.createContextualFragment(chunk);
                        target.appendChild(fragment);

                        // 自動捲動
                        reportPreview.contentWindow.scrollTo(0, iframeDoc.body.scrollHeight);
                    }
                }
            }

            statusBox.textContent = '分析完成！';
            statusBox.style.color = '#28a745';
        } catch (err) {
            statusBox.textContent = '分析失敗: ' + err.message;
            statusBox.style.color = '#dc3545';
            reportPreview.srcdoc = `<div style='padding:20px; color:red; font-family:sans-serif;'><h2>⚠️ 分析出錯</h2><p>${escapeHtml(err.message)}</p></div>`;
        } finally {
            setButtonsDisabled(false);
        }
    });

    // 刪除原本的 pollAnalysisStatus 函數，因為現在是用串流直接處理
});
