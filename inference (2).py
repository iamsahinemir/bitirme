# inference.py

import re
import pandas as pd
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Static QA + retrieval için
from rag_utils import df, idx_q, qa_map, embedder_q, rag_answer as _rag_answer, extract_date

# ─────────────────────────────────────────────────────────────────────────────
# 0️⃣ Asistan meta‐bilgileri
ASSISTANT_NAME    = "MakineTitreşimAsistanı"
ASSISTANT_PURPOSE = (
    "Bu asistanın amacı, endüstriyel makinelerin titreşim verilerini "
    "analiz ederek sorularınıza doğrudan ve anlaşılır cevaplar vermektir."
)
SYSTEM_PREFIX = (
    f"Asistan: {ASSISTANT_NAME}\n"
    f"Amacı: {ASSISTANT_PURPOSE}\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ Tokenizer’ı yükle
tokenizer = AutoTokenizer.from_pretrained(
    "iamsahinemir/meta-llama",
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣ Base model 8-bit olarak yükle
bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
base_model = AutoModelForCausalLM.from_pretrained(
    "iamsahinemir/meta-llama",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣ LoRA adapter’ı ekle
model = PeftModel.from_pretrained(
    base_model,
    "iamsahinemir/bitirme_model_lora",
    device_map="auto"
)
model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# 4️⃣ Satır-temelli FAISS index — dynamic context için
row_texts = df.apply(
    lambda r: f"Tarih: {r['date']}, Durum: {r['Situation']}, Değer: {r['Value']:.2f}",
    axis=1
).tolist()
row_embs = embedder_q.encode(row_texts, convert_to_numpy=True)
faiss.normalize_L2(row_embs)
row_idx = faiss.IndexFlatIP(row_embs.shape[1])
row_idx.add(row_embs)

# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣ generate_answer: app.py’in çağıracağı fonksiyon
def generate_answer(user_question: str) -> str:
    # (1) normalize “makine” → “RTF makinesi”
    q_norm = re.sub(r"\bmakine\b", "RTF makinesi", user_question, flags=re.IGNORECASE)

    # (2) veri‐ilgili değilse → direkt LLM
    if not re.search(r"\b(makine|titreşim|alarm|rtf)\b", q_norm, flags=re.IGNORECASE):
        prompt = SYSTEM_PREFIX + "\n" + f"Soru: {q_norm}\nCevap:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out    = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    # (3a) veri‐ilgili ise önce static QA
    date = extract_date(q_norm)
    ans  = _rag_answer(
        q_norm,
        df,
        model=None,
        tokenizer=None,
        threshold=0.40,
        date=date
    )

    # (3b) fallback
    if any(tok in ans for tok in ["Cevap bulunamadı", "Lütfen sorunuzda", "Tam olarak anlayamadım"]):
        ue = embedder_q.encode([q_norm], convert_to_numpy=True)
        faiss.normalize_L2(ue)
        D_rows, I_rows = row_idx.search(ue, 5)
        context = "\n".join(row_texts[i] for i in I_rows[0])

        prompt = (
            SYSTEM_PREFIX
            + "\nAşağıda RTF makinesi titreşim verileri var:\n"
            f"{context}\n\n"
            f"Soru: {q_norm}\n"
            "Bu verilere dayanarak cevap verin:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out    = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    # (3c) static QA cevabı
    return ans
