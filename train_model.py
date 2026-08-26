import torch
import os
import sys
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import multiprocessing

# 確保載入 config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from models import Vocab, LegalAutoencoder

# --- 定義非監督式 Dataset ---
class UnsupervisedLegalDataset(Dataset):
    def __init__(self, texts, vocab, max_len=512):
        self.texts = texts
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 目標是自己預測自己 (Autoencoder)
        indices = self.vocab.encode(self.texts[idx], self.max_len)
        tensor_data = torch.tensor(indices, dtype=torch.long)
        return tensor_data, tensor_data # 輸入與目標相同

def train_unsupervised():
    print("--- 啟動非監督式特徵學習流程 (自由分布模式) ---")

    # 1. 讀取數據
    dataset_path = os.path.join(config.BASE_DIR, "dataset", "legal_dataset.csv")
    if not os.path.exists(dataset_path):
        print("錯誤：找不到 legal_dataset.csv，請先運行標註或爬蟲腳本。")
        return None

    df = pd.read_csv(dataset_path)
    texts = df['TextContent'].fillna("").tolist()
    if not texts:
        print("錯誤：legal_dataset.csv 沒有可用文本，訓練終止。")
        return None

    # 2. 建立辭典與 Dataset
    vocab = Vocab(texts, max_vocab=config.TRAIN_MAX_VOCAB_SIZE)
    dataset = UnsupervisedLegalDataset(texts, vocab, max_len=config.TRAIN_MAX_SEQ_LEN)

    # 根據 config 決定 Batch Size，若為 0 則一次讀取所有
    batch_size = config.TRAIN_BATCH_SIZE if config.TRAIN_BATCH_SIZE > 0 else len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. 初始化模型與優化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備：{device}")

    model = LegalAutoencoder(len(vocab.vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 忽略 Padding
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN_LEARNING_RATE)

    # 4. 訓練迴圈
    epochs = config.TRAIN_EPOCHS
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)

            # Reshape 為 [batch * seq_len, vocab_size] 以符合 CrossEntropyLoss
            loss = criterion(logits.view(-1, len(vocab.vocab)), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"==> Epoch {epoch+1} 完成，平均 Loss: {total_loss/len(dataloader):.4f}")

    # 5. 儲存模型
    save_path = os.path.join(config.BASE_DIR, "dataset", "best_model.pth")
    # 同時儲存詞典資訊方便未來推論使用
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab.vocab,
        'inv_vocab': vocab.inv_vocab
    }
    torch.save(checkpoint, save_path)
    print(f"訓練完成！特徵提取模型已儲存至：{save_path}")
    return save_path

if __name__ == "__main__":
    # Windows Multiprocessing 修正
    multiprocessing.freeze_support()
    sys.exit(0 if train_unsupervised() else 1)
