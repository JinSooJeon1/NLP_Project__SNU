!pip install datasets
!mkdir essays
import re
import pickle
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from datasets import load_dataset
import essays

# --------------------------------------------------------
#  Path 
# --------------------------------------------------------
OUT_DIR = Path(
    "/content/drive/MyDrive/NLP_Personality_Prediction/personality-prediction-from-text/data/essays"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 입력 데이터 경로 (git repo 기준 상대 경로)
ESSAYS_CSV = Path("data/training/essays.csv")
KAGGLE_MBTI_CSV = Path("data/training/mbti_1.csv")
REDDIT_MBTI_CSV = Path("data/training/typed_comments.csv")
EMOTION_LEXICON_CSV = Path("data/training/Emotion_Lexicon.csv")

# Emotion Lexicon 옵션
USE_EMOTION_FILTER_FOR_BIG5 = False   # True면 Big5 텍스트에서 감정 문장만 남김


# --------------------------------------------------------
#  Utility Functions
# --------------------------------------------------------
def _to01(x: Any) -> int:
    """라벨을 0/1 정수로 안전 변환"""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "y", "yes", "true", "t"}:
            return 1
        if s in {"0", "n", "no", "false", "f"}:
            return 0
    if isinstance(x, bool):
        return int(x)
    return 0  # 나머지는 0으로


def _load_emotion_words(path: Path) -> Optional[List[str]]:
    """Emotion_Lexicon.csv를 읽어 감정 단어 리스트 반환"""
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    # 모든 감정 열이 0인 단어 제거
    df = df[(df.T != 0).any()]
    return df.index.tolist()


def _sentence_split(text: str) -> List[str]:
    """아주 단순한 영어 문장 분리기"""
    return re.split(r'(?<=[.!?])\s+', text.strip())


def remove_unemotional_sentences(emotional_words: List[str], text_as_one_string: str) -> str:
    """감정 단어가 없는 문장은 제거하고 감정 문장만 이어붙임"""
    reduced_s = ""
    sentences = _sentence_split(text_as_one_string)
    emo_set = set(w.lower() for w in emotional_words)
    for s in sentences:
        toks = re.findall(r"\b[\w']+\b", s.lower())
        if any(t in emo_set for t in toks):
            reduced_s += s + " "
    return reduced_s.strip() if reduced_s else text_as_one_string


# --------------------------------------------------------
#  Big5: essay.Essay 객체 생성 유틸
# --------------------------------------------------------
def create_essays(df: pd.DataFrame,
                  emotional_words: Optional[List[str]] = None,
                  use_filter: bool = False) -> List["essay.Essay"]:
    """
    df: 반드시 [TEXT, cEXT, cNEU, cAGR, cCON, cOPN] 컬럼 포함
    emotional_words: Emotion Lexicon 단어 리스트
    use_filter: True이면 Emotion Lexicon 기반 감정문장만 남김
    """
    essays_list = []
    for _, row in df.iterrows():
        text = str(row["TEXT"])

        clean_text = re.sub(r"\s+", " ", text).strip()
        filtered_text = clean_text

        if use_filter and emotional_words is not None:
            filtered_text = remove_unemotional_sentences(emotional_words, clean_text)

        e = essay.Essay(
            filtered_text if use_filter else clean_text,
            row["cEXT"],
            row["cNEU"],
            row["cAGR"],
            row["cCON"],
            row["cOPN"],
        )

        e.clean_text = clean_text
        e.filtered_text = filtered_text
        essays_list.append(e)

    return essays_list


# --------------------------------------------------------
#  Big5 소스 1: Hugging Face jingjietan/pandora-big5
# --------------------------------------------------------
def load_big5_from_hf() -> pd.DataFrame:
    """
    HF 'jingjietan/pandora-big5' → [TEXT, cEXT, cNEU, cAGR, cCON, cOPN]
    """
    print("[Big5] Loading from HuggingFace: jingjietan/pandora-big5 ...")
    ds = load_dataset("jingjietan/pandora-big5")

    parts = []
    for split in ("train", "validation", "test"):
        if split in ds:
            parts.append(ds[split].to_pandas())
    df = pd.concat(parts, ignore_index=True)

    # text, O, C, E, A, N → TEXT, cOPN, cCON, cEXT, cAGR, cNEU
    df = df.rename(columns={"text": "TEXT"})
    df["cOPN"] = df["O"].map(_to01)
    df["cCON"] = df["C"].map(_to01)
    df["cEXT"] = df["E"].map(_to01)
    df["cAGR"] = df["A"].map(_to01)
    df["cNEU"] = df["N"].map(_to01)

    df = df[["TEXT", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]].copy()
    df["TEXT"] = df["TEXT"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    print(f"[Big5] HF rows: {len(df)}")
    return df


# --------------------------------------------------------
#  Big5 소스 2: data/training/essays.csv (원래 Essays Big5)
# --------------------------------------------------------
def load_big5_from_essays_csv(path: Path) -> pd.DataFrame:
    """
    essays.csv (TEXT, cEXT~cOPN, 'y'/'n' 또는 0/1) → [TEXT, cEXT, cNEU, cAGR, cCON, cOPN]
    """
    if not path.exists():
        print(f"[Big5] essays.csv not found at {path}, 이 소스는 건너뜁니다.")
        return pd.DataFrame(columns=["TEXT", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"])

    print(f"[Big5] Loading essays.csv from {path} ...")
    df = pd.read_csv(path, encoding="cp1252", delimiter=",", quotechar='"')

    # 원래 코드 구조: TEXT, cEXT, cNEU, cAGR, cCON, cOPN 존재 가정
    trait_cols = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
    if "TEXT" not in df.columns:
        raise ValueError("essays.csv 파일에 'TEXT' 컬럼이 필요합니다.")

    for col in trait_cols:
        if col not in df.columns:
            raise ValueError(f"essays.csv 파일에 '{col}' 컬럼이 필요합니다.")
        df[col] = df[col].replace({"n": "0", "y": "1"})
        df[col] = df[col].map(_to01)

    df = df[["TEXT", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]].copy()
    df["TEXT"] = df["TEXT"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    print(f"[Big5] essays.csv rows: {len(df)}")
    return df


# --------------------------------------------------------
#  MBTI 소스 1: Kaggle mbti_1.csv (MBTI 그대로)
# --------------------------------------------------------
def load_mbti_from_kaggle(path: Path) -> pd.DataFrame:
    """
    Kaggle 'mbti_1.csv' → [TEXT, MBTI]
    - posts: 긴 텍스트, '|||' 로 구분 → ' '로 치환
    - type: MBTI 문자열 (예: INTJ, ENFP ...)
    """
    if not path.exists():
        print(f"[MBTI] Kaggle mbti_1.csv not found at {path}, 이 소스는 건너뜁니다.")
        return pd.DataFrame(columns=["TEXT", "MBTI"])

    print(f"[MBTI] Loading Kaggle mbti_1.csv from {path} ...")
    df = pd.read_csv(path)

    if "posts" not in df.columns or "type" not in df.columns:
        raise ValueError("mbti_1.csv에는 'posts'와 'type' 컬럼이 필요합니다.")

    df_mbti = df[["posts", "type"]].copy()
    df_mbti = df_mbti.rename(columns={"posts": "TEXT", "type": "MBTI"})
    # "|||" 제거
    df_mbti["TEXT"] = df_mbti["TEXT"].astype(str).str.replace("|||", " ")
    df_mbti["TEXT"] = df_mbti["TEXT"].str.replace(r"\s+", " ", regex=True).str.strip()

    print(f"[MBTI] Kaggle rows: {len(df_mbti)}")
    return df_mbti


# --------------------------------------------------------
#  MBTI 소스 2: Reddit typed_comments.csv (MBTI 그대로)
# --------------------------------------------------------
def load_mbti_from_reddit(path: Path, min_word_count: int = 420) -> pd.DataFrame:
    """
    Reddit typed_comments.csv → [TEXT, MBTI]
    - comment: 텍스트
    - type: MBTI
    - word_count: 없으면 여기서 계산
    - min_word_count: 그 이상인 샘플만 사용
    """
    if not path.exists():
        print(f"[MBTI] Reddit typed_comments.csv not found at {path}, 이 소스는 건너뜁니다.")
        return pd.DataFrame(columns=["TEXT", "MBTI"])

    print(f"[MBTI] Loading Reddit typed_comments.csv from {path} ...")
    df = pd.read_csv(path)

    if "comment" not in df.columns or "type" not in df.columns:
        raise ValueError("typed_comments.csv에는 'comment'와 'type' 컬럼이 필요합니다.")

    # word_count 없으면 계산
    if "word_count" not in df.columns:
        df["word_count"] = df["comment"].astype(str).str.split().str.len()

    df = df[df["word_count"] > min_word_count].copy()

    df_mbti = df[["comment", "type"]].copy()
    df_mbti = df_mbti.rename(columns={"comment": "TEXT", "type": "MBTI"})
    df_mbti["TEXT"] = df_mbti["TEXT"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    print(f"[MBTI] Reddit rows (after word_count>{min_word_count}): {len(df_mbti)}")
    return df_mbti


# --------------------------------------------------------
#  Main Pipeline
# --------------------------------------------------------
def main():
    print("=== Personality Preprocessing (Big5 + MBTI) ===")

    # 1) Emotion Lexicon (Big5용 옵션)
    emotional_words = _load_emotion_words(EMOTION_LEXICON_CSV) if EMOTION_LEXICON_CSV.exists() else None
    if emotional_words is None:
        print("[INFO] Emotion Lexicon not found or empty -> 감정문장 필터 비활성화")
    else:
        print(f"[INFO] Emotion Lexicon loaded: {len(emotional_words)} words")

    # ------------------- Big5 파이프라인 -------------------
    print("\n[STEP] Building Big5 dataset (HF + essays.csv) ...")

    df_big5_hf = load_big5_from_hf()
    df_big5_essays = load_big5_from_essays_csv(ESSAYS_CSV)

    # 소스별로 따로 쓰고 싶으면 각각 저장해도 되고,
    # 여기서는 하나로 합쳐서 Big5 전체 데이터로 사용
    df_big5_all = pd.concat([df_big5_hf, df_big5_essays], ignore_index=True)
    print(f"[Big5] Total rows (HF + essays.csv): {len(df_big5_all)}")

    big5_essays = create_essays(
        df_big5_all,
        emotional_words=emotional_words,
        use_filter=USE_EMOTION_FILTER_FOR_BIG5 and (emotional_words is not None),
    )

    # Big5 저장 (동적 파일명 + 고정 파일명)
    n_big5 = len(big5_essays)
    big5_dynamic = OUT_DIR / f"essays_big5_{n_big5}.p"
    big5_fixed = OUT_DIR / "essays_big5.p"

    with open(big5_dynamic, "wb") as f:
        pickle.dump(big5_essays, f)
    with open(big5_fixed, "wb") as f:
        pickle.dump(big5_essays, f)

    print(f"[OK] Big5 essays saved: n={n_big5}")
    print(f"[OK] Big5 dynamic file: {big5_dynamic}")
    print(f"[OK] Big5 fixed   file: {big5_fixed}")

    # ------------------- MBTI 파이프라인 -------------------
    print("\n[STEP] Building MBTI dataset (Kaggle + Reddit) ...")

    df_mbti_kaggle = load_mbti_from_kaggle(KAGGLE_MBTI_CSV)
    df_mbti_reddit = load_mbti_from_reddit(REDDIT_MBTI_CSV)

    df_mbti_all = pd.concat([df_mbti_kaggle, df_mbti_reddit], ignore_index=True)
    print(f"[MBTI] Total rows (Kaggle + Reddit): {len(df_mbti_all)}")

    # MBTI 저장 (DataFrame 그대로 pickle)
    n_mbti = len(df_mbti_all)
    mbti_dynamic = OUT_DIR / f"text_mbti_{n_mbti}.p"
    mbti_fixed = OUT_DIR / "text_mbti.p"

    with open(mbti_dynamic, "wb") as f:
        pickle.dump(df_mbti_all, f)
    with open(mbti_fixed, "wb") as f:
        pickle.dump(df_mbti_all, f)

    print(f"[OK] MBTI text saved: n={n_mbti}")
    print(f"[OK] MBTI dynamic file: {mbti_dynamic}")
    print(f"[OK] MBTI fixed   file: {mbti_fixed}")

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
