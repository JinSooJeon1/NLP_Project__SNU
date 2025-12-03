# ============================================
# 0. 환경 설정 & Imports
# ============================================
%cd /content/drive/MyDrive/NLP_Personality_Prediction/personality-prediction-from-text

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import essay  # 원래 repo에서 쓰는 모듈 (구조만 유지)
import os

print("PyTorch version:", torch.__version__)

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================
# 1. GloVe + MeanEmbeddingVectorizer
#    (model_glove.ipynb 전처리 스타일)
# ============================================

class MeanEmbeddingVectorizer(object):
    """
    model_glove.ipynb와 동일 컨셉:
    X의 각 element(essay)는 e.glove(dict: word -> vector)라고 가정.
    dict를 순회하면 key가 word가 되므로 그대로 사용 가능.
    """
    def __init__(self, word2vec: dict):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            # 아무 키나 하나 꺼내서 차원 확인
            first_key = next(iter(word2vec.keys()))
            self.dim = len(word2vec[first_key])
        else:
            self.dim = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: iterable of essay.glove (dict 또는 list of tokens)
        결과: [N, dim] numpy array
        """
        vectors = []
        for words in X:
            # words가 dict이면 key가 token, list면 그대로 token
            if isinstance(words, dict):
                tokens = list(words.keys())
            else:
                tokens = list(words)
            vecs = [self.word2vec[w] for w in tokens if w in self.word2vec]
            if len(vecs) == 0:
                vectors.append(np.zeros(self.dim, dtype=np.float32))
            else:
                vectors.append(np.mean(vecs, axis=0))
        return np.array(vectors, dtype=np.float32)


def build_glove_for_essays(essays, glove_path_small: str):
    """
    model_glove.ipynb 로직을 함수로 정리:
    1) GloVe 파일 로드 → df (word + vector)
    2) 모든 essay의 e.words를 corpus로 모음
    3) corpus와 GloVe를 inner join해서 df_mywords
    4) glove_mywords dict(word -> vector) 생성
    5) 각 essay에 e.glove dict(word -> vector) 추가
    """
    # 1) GloVe 파일 로드
    print("Loading GloVe from:", glove_path_small)
    df_glove = pd.read_csv(glove_path_small, sep=" ", quoting=3, header=None, encoding="utf-8")
    # 첫 col: 단어, 나머지: 벡터

    # 2) 코퍼스 만들기
    corpus = []
    for e in essays:
        for w in e.words:
            corpus.append(w)
    df_corpus = pd.DataFrame(corpus)  # 한 컬럼짜리 (0열에 단어)

    # 3) inner join (우리 코퍼스 단어 ∩ GloVe 단어)
    df_mywords = df_glove.merge(df_corpus, left_on=0, right_on=0)
    df_mywords = df_mywords.drop_duplicates()

    # 4) dict(word -> vector)
    df_temp = df_mywords.set_index(0)
    glove_mywords = {
        key: val.values.astype(np.float32)
        for key, val in df_temp.T.items()
    }  # key: word, val: np.array(dim,)

    print(f"GloVe filtered to {len(glove_mywords)} words in corpus.")

    # 5) 각 essay에 e.glove 딕셔너리 생성
    for e in essays:
        df_temp_e = pd.DataFrame(e.words)
        try:
            df_temp_e = df_temp_e.merge(df_mywords, left_on=0, right_on=0)
            df_temp_e = df_temp_e.drop_duplicates()
            df_temp_e = df_temp_e.set_index(0)
            e.glove = {
                key: val.values.astype(np.float32)
                for key, val in df_temp_e.T.items()
            }
        except Exception as ex:
            print("glove building error on sample:", ex)
            e.glove = {}

    return glove_mywords, essays


# ============================================
# 2. Dataset 정의 (Big5 / MBTI)
# ============================================

class Big5EssayDataset(Dataset):
    """
    Big5 essay dataset:
      - 입력: e.glove → MeanEmbeddingVectorizer → [dim] 벡터
      - 라벨: [cEXT, cNEU, cAGR, cCON, cOPN]
    """
    def __init__(self, essays, glove_vec_dict):
        self.essays = essays
        self.vectorizer = MeanEmbeddingVectorizer(glove_vec_dict)

        self.X = self.vectorizer.transform([e.glove for e in essays])  # [N, dim]
        self.y = np.stack(
            [[e.cEXT, e.cNEU, e.cAGR, e.cCON, e.cOPN] for e in essays],
            axis=0
        ).astype(np.float32)

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])      # [dim]
        y = torch.from_numpy(self.y[idx])      # [5]
        return x, y


class MBTIDataset(Dataset):
    """
    MBTI 텍스트 데이터셋:
      - samples: 객체 리스트 (s.words, s.mbti_vec(4D binary: EI,SN,TF,JP))
      - build_glove_for_essays로 s.glove(dict: word -> vector)가 이미 생성되어 있다고 가정.
    """
    def __init__(self, samples, glove_vec_dict):
        self.samples = samples
        self.vectorizer = MeanEmbeddingVectorizer(glove_vec_dict)
        self.X = self.vectorizer.transform([s.glove for s in samples])  # [N, dim]
        # mbti_vec: [EI,SN,TF,JP] 0/1, E/N/F/J=1, I/S/T/P=0
        self.y = np.stack([s.mbti_vec for s in samples], axis=0).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])      # [dim]
        y = torch.from_numpy(self.y[idx])      # [4]
        return x, y


# ============================================
# 3. CorrGraph-MTL 모델 (GloVe mean → small encoder → CGC)
# ============================================

class SmallEncoder(nn.Module):
    """
    GloVe mean vector -> hidden representation
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        # x: [B, in_dim]
        return self.net(x)  # [B, hidden_dim]


class CorrGraphMTLFromGlove(nn.Module):
    """
    입력:
      big5_vec: [B, D_glove]    (Big5 텍스트의 GloVe mean)
      mbti_vec: [B, D_glove]    (MBTI 텍스트의 GloVe mean)

    처리:
      1) 각 벡터를 SmallEncoder 통과 → z_b, z_m
      2) z_b → 5D proto, z_m → 4D proto
      3) [5+4=9D] y_pre에 대해 CGC(message passing)
      4) 최종 head로 Big5 회귀, MBTI 로짓 출력
    """
    def __init__(
        self,
        glove_dim: int,
        hidden_dim: int = 256,
        head_hidden: int = 64,
        rank_correction: int = 2,
        msg_steps: int = 1,
        init_r=(0.65, 0.50, 0.40, 0.40)  # (EI–E, SN–O, TF–A, JP–C)
    ):
        super().__init__()

        self.msg_steps = msg_steps

        # encoders
        self.big5_enc = SmallEncoder(glove_dim, hidden_dim)
        self.mbti_enc = SmallEncoder(glove_dim, hidden_dim)

        # proto projection
        self.big5_proj = nn.Linear(hidden_dim, 5)  # E,N,A,C,O
        self.mbti_proj = nn.Linear(hidden_dim, 4)  # EI,SN,TF,JP  (1 = E/N/F/J)

        # --- Correlation adjacency (A0) ---
        A0 = torch.zeros(9, 9)
        r_ei_e, r_sn_o, r_tf_a, r_jp_c = init_r
        # indexes: 0:E,1:N,2:A,3:C,4:O, 5:EI,6:SN,7:TF,8:JP
        A0[0, 5] = A0[5, 0] = r_ei_e      # E <-> EI
        A0[4, 6] = A0[6, 4] = r_sn_o      # O <-> SN
        A0[2, 7] = A0[7, 2] = r_tf_a      # A <-> TF
        A0[3, 8] = A0[8, 3] = r_jp_c      # C <-> JP
        # N (index 1)는 연결 없음

        with torch.no_grad():
            D = torch.diag(A0.sum(dim=1).clamp(min=1e-6))
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
            A0_norm = D_inv_sqrt @ A0 @ D_inv_sqrt

        self.register_buffer("A0", A0)
        self.register_buffer("A0_norm", A0_norm)

        # 저랭크 보정 ΔA
        self.U = nn.Parameter(torch.zeros(9, rank_correction))
        self.V = nn.Parameter(torch.zeros(9, rank_correction))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

        # message passing strength
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # final heads
        self.big5_head = nn.Sequential(
            nn.Linear(5, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 5)      # regression
        )
        self.mbti_head = nn.Sequential(
            nn.Linear(4, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 4)      # logits
        )

    def _message_passing(self, y):
        """
        y: [B,9]
        y_out = y + alpha * A y (1~T step residual)
        """
        A = self.A0_norm + self.U @ self.V.t()
        y_msg = y
        for _ in range(self.msg_steps):
            y_msg = y_msg + self.alpha * (y_msg @ A.t())
        return y_msg

    def graph_smoothness_loss(self, y_cgc):
        """
        Edges: (E,EI), (O,SN), (A,TF), (C,JP)
        """
        edges = [(0,5), (4,6), (2,7), (3,8)]
        loss = 0.0
        for (i, j) in edges:
            w = self.A0[i, j].abs()
            if w > 0:
                diff = (y_cgc[:, i] - y_cgc[:, j])**2
                loss = loss + w * diff.mean()
        return loss

    def forward(self, big5_vec, mbti_vec):
        """
        big5_vec, mbti_vec: [B, glove_dim] float tensor
        """
        zb = self.big5_enc(big5_vec)    # [B, hidden]
        zm = self.mbti_enc(mbti_vec)    # [B, hidden]

        b_proto = self.big5_proj(zb)    # [B,5]
        m_proto = self.mbti_proj(zm)    # [B,4]

        # MBTI는 sigmoid로 0~1로 정규화 후 concat
        y_pre = torch.cat([b_proto, torch.sigmoid(m_proto)], dim=-1)  # [B,9]

        y_cgc = self._message_passing(y_pre)

        b_cgc = y_cgc[:, :5]
        m_cgc = y_cgc[:, 5:]

        big5_pred = self.big5_head(b_cgc)       # [B,5]
        mbti_logits = self.mbti_head(m_cgc)     # [B,4]

        return {
            "big5_pred": big5_pred,
            "mbti_logits": mbti_logits,
            "y_pre": y_pre,
            "y_cgc": y_cgc,
        }

    def compute_losses(
        self,
        out,
        big5_gold=None,   # [B,5]
        mbti_gold=None,   # [B,4]
        w_graph: float = 0.2
    ):
        L = 0.0
        logs = {}

        if big5_gold is not None:
            L_big5 = F.huber_loss(out["big5_pred"], big5_gold, delta=1.0)
            L = L + L_big5
            logs["L_big5"] = float(L_big5.item())

        if mbti_gold is not None:
            L_mbti = F.binary_cross_entropy_with_logits(out["mbti_logits"], mbti_gold)
            L = L + L_mbti
            logs["L_mbti"] = float(L_mbti.item())

        L_graph = self.graph_smoothness_loss(out["y_cgc"])
        L = L + w_graph * L_graph
        logs["L_graph"] = float(L_graph.item())
        logs["L_total"] = float(L.item())
        return L, logs


# ============================================
# 4. Big5 + MBTI 데이터 로드 & GloVe 구축
# ============================================

# 1) Big5 essays 로드
ESSAY_PATH = "/content/drive/MyDrive/NLP_Personality_Prediction/personality-prediction-from-text/data/essays/essays_hf2467.p"
assert os.path.exists(ESSAY_PATH), f"Essay file not found: {ESSAY_PATH}"

essays = pickle.load(open(ESSAY_PATH, "rb"))
print("Number of Big5 essays:", len(essays))

# 2) MBTI 샘플 로드
#   - 각 sample은 최소한 아래 속성을 가진다고 가정:
#       s.words: 토큰 리스트 (list[str])
#       s.mbti_vec: shape (4,) 의 0/1 numpy array (EI, SN, TF, JP)
MBTI_PATH = "/content/drive/MyDrive/NLP_Personality_Prediction/personality-prediction-from-text/data/essays/essays_kaggle8675.p"
assert os.path.exists(MBTI_PATH), f"MBTI file not found: {MBTI_PATH}"

mbti_samples = pickle.load(open(MBTI_PATH, "rb"))
print("Number of MBTI samples:", len(mbti_samples))

# 3) GloVe 경로 (Stanford GloVe 50d)
GLOVE_SMALL = "data/pretrained/glove.6B.50d.txt"
assert os.path.exists(GLOVE_SMALL), f"GloVe file not found: {GLOVE_SMALL}"

# 4) Big5 + MBTI 전체 코퍼스에 대해 한 번에 GloVe 필터링
#    → essays, mbti_samples 둘 다 .glove dict가 채워짐
all_texts = essays + mbti_samples
glove_mywords, _ = build_glove_for_essays(all_texts, GLOVE_SMALL)

# 5) Big5 / MBTI 각각 Train/Test split
train_essays, test_essays = train_test_split(
    essays, test_size=0.2, random_state=42
)

train_mbti, test_mbti = train_test_split(
    mbti_samples, test_size=0.2, random_state=42
)

train_ds_big5 = Big5EssayDataset(train_essays, glove_mywords)
test_ds_big5  = Big5EssayDataset(test_essays, glove_mywords)

train_ds_mbti = MBTIDataset(train_mbti, glove_mywords)
test_ds_mbti  = MBTIDataset(test_mbti, glove_mywords)

train_loader_big5 = DataLoader(train_ds_big5, batch_size=32, shuffle=True,  drop_last=False)
test_loader_big5  = DataLoader(test_ds_big5,  batch_size=32, shuffle=False, drop_last=False)

train_loader_mbti = DataLoader(train_ds_mbti, batch_size=32, shuffle=True,  drop_last=False)
test_loader_mbti  = DataLoader(test_ds_mbti,  batch_size=32, shuffle=False, drop_last=False)

glove_dim = train_ds_big5.X.shape[1]
print("GloVe dim:", glove_dim)


# ============================================
# 5. 모델 및 옵티마이저
# ============================================

model = CorrGraphMTLFromGlove(glove_dim=glove_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ============================================
# 6. 평가 함수 (Big5 / MBTI)
# ============================================

def evaluate_big5(model, data_loader):
    model.eval()
    mae_sum = 0.0
    n = 0
    with torch.no_grad():
        for big5_x, big5_y in data_loader:
            big5_x = big5_x.to(device, non_blocking=True)
            big5_y = big5_y.to(device, non_blocking=True)

            # MBTI 입력은 0벡터 (Big5 전용 평가)
            mbti_x = torch.zeros(big5_x.size(0), glove_dim, device=device)

            out = model(big5_x, mbti_x)
            pred = out["big5_pred"]  # [B,5]
            mae_sum += torch.abs(pred - big5_y).sum(dim=0)  # trait-wise sum
            n += big5_x.size(0)
    mae = mae_sum / n
    return mae.cpu().numpy()  # [5]


def evaluate_mbti(model, data_loader, threshold=0.5):
    """
    MBTI는 4개 축(EI, SN, TF, JP) 각각에 대해
    BCE loss와 간단한 accuracy를 리턴
    """
    model.eval()
    bce_sum = 0.0
    n = 0
    correct = torch.zeros(4, device=device)  # trait-wise accuracy용

    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    with torch.no_grad():
        for mbti_x, mbti_y in data_loader:
            mbti_x = mbti_x.to(device, non_blocking=True)  # [B, D]
            mbti_y = mbti_y.to(device, non_blocking=True)  # [B, 4]

            # Big5 입력은 0벡터 (MBTI 전용 평가)
            big5_x = torch.zeros(mbti_x.size(0), glove_dim, device=device)

            out = model(big5_x, mbti_x)
            logits = out["mbti_logits"]  # [B,4]

            bce_sum += bce_loss_fn(logits, mbti_y).item()
            n += mbti_x.size(0)

            preds = (torch.sigmoid(logits) >= threshold).float()
            correct += (preds == mbti_y).float().sum(dim=0)

    avg_bce = bce_sum / n
    acc = (correct / n).cpu().numpy()  # [4]
    return avg_bce, acc


# ============================================
# 7. 멀티태스크 학습 루프 (Big5 + MBTI 동시 학습)
# ============================================

n_epochs = 5
lambda_mbti = 1.0      # MBTI loss 가중치 (원하면 조절)
w_graph = 0.2          # graph smoothness loss 가중치

for epoch in range(1, n_epochs + 1):
    model.train()

    mbti_iter = iter(train_loader_mbti)

    for big5_x, big5_y in train_loader_big5:
        big5_x = big5_x.to(device, non_blocking=True)  # [B, D]
        big5_y = big5_y.to(device, non_blocking=True)  # [B, 5]

        # MBTI 배치를 하나 끌어옴. 다 돌면 다시 iterator 리셋.
        try:
            mbti_x, mbti_y = next(mbti_iter)
        except StopIteration:
            mbti_iter = iter(train_loader_mbti)
            mbti_x, mbti_y = next(mbti_iter)

        mbti_x = mbti_x.to(device, non_blocking=True)  # [B2, D]
        mbti_y = mbti_y.to(device, non_blocking=True)  # [B2, 4]

        # ---- Forward & Loss (멀티태스크) ----
        # Big5 / MBTI 배치 크기가 다를 수 있으므로
        # 여기서는 단순히 "같은 크기"로 맞추기 위해
        # 작은 쪽 batch size로 잘라서 사용
        B = min(big5_x.size(0), mbti_x.size(0))
        big5_x_batch = big5_x[:B]
        big5_y_batch = big5_y[:B]
        mbti_x_batch = mbti_x[:B]
        mbti_y_batch = mbti_y[:B]

        out = model(big5_x_batch, mbti_x_batch)

        # 전체 loss = L_big5 + λ * L_mbti + w_graph * L_graph
        L_big5 = F.huber_loss(out["big5_pred"], big5_y_batch, delta=1.0)
        L_mbti = F.binary_cross_entropy_with_logits(out["mbti_logits"], mbti_y_batch)
        L_graph = model.graph_smoothness_loss(out["y_cgc"])

        loss = L_big5 + lambda_mbti * L_mbti + w_graph * L_graph

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ----- Epoch 끝에서 평가 -----
    mae_big5 = evaluate_big5(model, test_loader_big5)
    bce_mbti, acc_mbti = evaluate_mbti(model, test_loader_mbti)

    print(f"[Epoch {epoch}] "
          f"L_big5={L_big5.item():.4f}, L_mbti={L_mbti.item():.4f}, L_graph={L_graph.item():.4f}")
    print(f"  Big5 val MAE (EXT,NEU,AGR,CON,OPN) = {mae_big5}")
    print(f"  MBTI val BCE = {bce_mbti:.4f}, "
          f"acc(EI,SN,TF,JP) = {acc_mbti}")
