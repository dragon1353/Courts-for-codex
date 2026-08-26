import torch
import torch.nn as nn
from collections import Counter

class Vocab:
    def __init__(self, texts_or_dict, inv_vocab=None, max_vocab=10000):
        if isinstance(texts_or_dict, dict):
            # 從 dictionary 載入 (Inference 模式)
            self.vocab = texts_or_dict
            self.inv_vocab = inv_vocab
        else:
            # 從文本建立 (Training 模式)
            all_text = "".join(str(t) for t in texts_or_dict)
            chars = [c for c in all_text if not c.isspace()]
            counts = Counter(chars)
            self.vocab = {char: i + 2 for i, (char, _) in enumerate(counts.most_common(max_vocab))}
            self.vocab['<PAD>'] = 0
            self.vocab['<UNK>'] = 1
            self.inv_vocab = {i: char for char, i in self.vocab.items()}

    def encode(self, text, max_len):
        indices = [self.vocab.get(c, 1) for c in str(text)]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return indices[:max_len]

class LegalAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super(LegalAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Encoder: 壓縮資訊
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Latent Space (中介特徵層)
        self.latent_layer = nn.Linear(hidden_dim * 2, hidden_dim)

        # Decoder: 嘗試還原資訊 (僅在訓練時用到)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        """只執行編碼器，供檢索與向量建置提取潛在特徵。"""
        # 1. Embedding 層
        embedded = self.embedding(x)

        # 2. Encoder: 透過雙向 LSTM 提取特徵
        _, (hn, _) = self.encoder(embedded)
        # 合併雙向最後一個 Hidden State 做為此段文本的特徵代表
        latent = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        return torch.relu(self.latent_layer(latent))

    def forward(self, x):
        """
        前向傳播：將輸入序列壓縮為潛在特徵 (Latent Feature)，再試圖還原。
        """
        latent = self.encode(x)

        # 3. Decoder: 展開並還原序列 (用於訓練計算 Loss)
        seq_len = x.size(1)
        z = latent.unsqueeze(1).repeat(1, seq_len, 1)

        decoded_out, _ = self.decoder_lstm(z)
        logits = self.output_layer(decoded_out)

        return logits, latent
