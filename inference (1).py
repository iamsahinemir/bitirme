# inference.py

import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rag_utils import df, idx_q, qa_map, embedder_q, rag_answer as _rag_answer

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ Türkçe ay adlarını normalize etmek için yardımcı fonksiyon
turkish_months = {
    "ocak":   "January",   "şubat":  "February", "mart":    "March",
    "nisan":  "April",     "mayıs":  "May",      "haziran": "June",
    "temmuz": "July",      "ağustos":"August",   "eylül":   "September",
    "ekim":   "October",   "kasım":  "November", "aralık":  "December"
}

def normalize_date_str(s: str) -> str | None:
    """
    Gelen ham metni temizler:
     - “/” veya “.” karakterlerini “-”e çevirir,
     - virgülleri çıkarır,
     - Türkçe ay adlarını İngilizce'ye çevirir,
     - dayfirst=True ile parse eder.
    """
    s_clean = s.strip().lower()
    # slash ve dot → dash
    s_clean = re.sub(r"[\/\.]", "-", s_clean)
    # comma kaldır
    s_clean = s_clean.replace(",", "")
    # Türkçe ay → İngilizce
    for tr, en in turkish_months.items():
        if tr in s_clean:
            s_clean = s_clean.replace(tr, en.lower())
            break
    dt = pd.to_datetime(s_clean, dayfirst=True, errors="coerce")
    return dt.date().isoformat() if not pd.isna(dt) else None

def extract_date(user_q: str) -> str | None:
    """
    1) YYYY[-/.]MM[-/.]DD veya DD[-/.]MM[-/.]YYYY desenini yakala
    2) Bulamazsa tüm metni normalize_date_str ile dene
    """
    m = re.search(
        r"(\d{4}[-\/\.]\d{2}[-\/\.]\d{2})|(\d{2}[-\/\.]\d{2}[-\/\.]\d{4})",
        user_q
    )
    if m:
        return normalize_date_str(m.group(0))
    return normalize_date_str(user_q)

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣ Tokenizer’ı yükle
tokenizer = AutoTokenizer.from_pretrained(
    "iamsahinemir/meta-llama",
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣ Base model 8-bit olarak yükle
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
base_model = AutoModelForCausalLM.from_pretrained(
    "iamsahinemir/meta-llama",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# ─────────────────────────────────────────────────────────────────────────────
# 4️⃣ LoRA adapter’ı ekle
model = PeftModel.from_pretrained(
    base_model,
    "iamsahinemir/bitirme_model_lora",
    device_map="auto"
)
model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣ generate_answer: app.py’in çağıracağı fonksiyon
def generate_answer(user_question: str) -> str:
    """
    1) Soru içindeki tekil “makine” kelimesini “RTF makinesi” ile değiştirir
    2) Soru içinden tarih çıkar (varsa)
    3) FAISS-RAG yoluyla cevap ara, eşleşme yetersizse LLM fallback
    """
    # (1) "makine" → "RTF makinesi"
    normalized_q = re.sub(
        r"\bmakine\b",
        "RTF makinesi",
        user_question,
        flags=re.IGNORECASE
    )

    # (2) Tarih çıkar
    date = extract_date(normalized_q)

    # (3) RAG + fallback
    return _rag_answer(
        normalized_q,
        df,
        model=model,
        tokenizer=tokenizer,
        threshold=0.65,
        date=date
    )
