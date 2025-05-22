# rag_utils.py

import pandas as pd
import re
import faiss
import numpy as np
import inspect
from sentence_transformers import SentenceTransformer
from dateutil import parser as date_parser

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ Veriyi yükle ve hazırlık
DF_PATH = "https://huggingface.co/datasets/iamsahinemir/vibration/resolve/main/vibration_df.csv"
df      = pd.read_csv(DF_PATH)

# dateutil.isoparse tüm ISO-8601 formatlarını (timezone dahil) yakalar
df['Timestamp'] = df['Timestamp'].apply(date_parser.isoparse)
# Eğer hâlâ tz-aware ise naive’a çevir
if pd.api.types.is_datetime64tz_dtype(df['Timestamp'].dtype):
    df['Timestamp'] = df['Timestamp'].dt.tz_convert(None)
df['date']      = df['Timestamp'].dt.date

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣ Tarih normalizasyonu / çıkarma
turkish_months = {
    "ocak":"January","şubat":"February","mart":"March","nisan":"April",
    "mayıs":"May","haziran":"June","temmuz":"July","ağustos":"August",
    "eylül":"September","ekim":"October","kasım":"November","aralık":"December"
}

def normalize_date_str(s: str) -> str | None:
    s_clean = re.sub(r"[\,\.]", "", s.strip().lower())
    for tr, en in turkish_months.items():
        if tr in s_clean:
            s_clean = s_clean.replace(tr, en.lower())
            break
    dt = pd.to_datetime(s_clean, dayfirst=True, errors="coerce")
    return dt.date().isoformat() if not pd.isna(dt) else None

def extract_date(user_q: str) -> str | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})|(\d{2}-\d{2}-\d{4})", user_q)
    if m:
        return normalize_date_str(m.group(0))
    return normalize_date_str(user_q)

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣ 32 QA fonksiyonları

def answer_q1(df):
    red = df[df["Situation"].str.upper()=="RED"].sort_values("Timestamp")
    segs, start, prev = [], red.iloc[0]["Timestamp"], red.iloc[0]["Timestamp"]
    for t in red["Timestamp"].iloc[1:]:
        if (t - prev) > pd.Timedelta(minutes=1):
            segs.append((start, prev)); start = t
        prev = t
    segs.append((start, prev))
    s,e = max(segs, key=lambda x: x[1]-x[0])
    return f"{s.strftime('%Y-%m-%d %H:%M')} ile {e.strftime('%Y-%m-%d %H:%M')} arasında"

def answer_q2(df):
    cnt = df[df["Situation"].str.upper()=="YELLOW"].groupby("date").size()
    return ", ".join(str(d) for d in cnt[cnt==cnt.max()].index)

def answer_q3(df):
    cnt = df[df["Situation"].str.upper()=="RED"].groupby("date").size()
    return ", ".join(str(d) for d in cnt[cnt==cnt.max()].index)

def answer_q4(df):
    v = df["Value"]
    return f"Değerler {v.min():.2f}–{v.max():.2f} mm/s arasında dalgalanmıştır."

def answer_q5(df):
    mask = (df["Timestamp"] >= "2023-01-01") & (df["Timestamp"] <= "2023-12-31")
    return str(df[mask & (df["Situation"].str.upper()=="RED")].shape[0])

def answer_q6(df):
    return "Green: 0–2.8 mm/s; Yellow: 2.8–11.2 mm/s; Orange: 11.2–14 mm/s; Red: 14+ mm/s"

def answer_q7(df):
    return "14+ mm/s"

def answer_q8(df, date):
    sub = df[df['date']==pd.to_datetime(date).date()]
    return f"{sub['Value'].min():.2f}–{sub['Value'].max():.2f} mm/s"

def answer_q9(df, date):
    sub = df[df['date']==pd.to_datetime(date).date()]
    return sub['Situation'].mode()[0]

def answer_q10(df, date):
    sub = df[df['date']==pd.to_datetime(date).date()]
    mn,mx,m = sub['Value'].min(), sub['Value'].max(), sub['Value'].mean()
    return f"{date} tarihinde min={mn:.2f}, max={mx:.2f}, ort={m:.2f} mm/s"

def answer_q11(df):
    mask = (df['date']>=pd.to_datetime("2023-01-01").date()) & (df['date']<=pd.to_datetime("2023-12-31").date())
    dates = df[mask & (df["Situation"].str.upper()=="ORANGE")].groupby("date").size()
    return ", ".join(str(d) for d in dates[dates>0].index)

def answer_q12(df):
    vals = df[df['Timestamp'].dt.year==2023]['Value']
    return f"2023 performansı: min={vals.min():.2f}, max={vals.max():.2f}, ort={vals.mean():.2f} mm/s"

def answer_q13(df):
    all_dates = pd.date_range(df['date'].min(), df['date'].max()).date
    missing = sorted(set(all_dates) - set(df['date'].unique()))
    return ", ".join(str(d) for d in missing)

def answer_q14(df):
    total = df.groupby('date').size()
    red   = df[df['Situation'].str.upper()=='RED'].groupby('date').size().reindex(total.index, fill_value=0)
    full  = red[ red == total ].index
    return ", ".join(str(d) for d in full)

def answer_q15(df):
    mins = df[df['Situation'].str.upper()=="GREEN"].shape[0]
    return f"{mins/60:.2f} saat ({mins} dakika)"

def answer_q16(df):
    recent = df[df['Timestamp'] >= df['Timestamp'].max() - pd.Timedelta(days=90)]
    return str(recent[recent["Situation"].str.upper()=="YELLOW"]['date'].nunique())

def answer_q17(df):
    recent = df[df['Timestamp'] >= df['Timestamp'].max() - pd.Timedelta(days=90)]
    return str(recent[recent["Situation"].str.upper()=="ORANGE"]['date'].nunique())

def answer_q18(df):
    recent = df[df['Timestamp'] >= df['Timestamp'].max() - pd.Timedelta(days=90)]
    return str(recent[recent["Situation"].str.upper()=="GREEN"]['date'].nunique())

def answer_q19(df):
    recent = df[df['Timestamp'] >= df['Timestamp'].max() - pd.Timedelta(days=90)]
    return str(recent[recent["Situation"].str.upper()=="RED"]['date'].nunique())

def answer_q20(df):
    s = df.sort_values("Timestamp")
    prev = s.iloc[0]['Situation'].upper()
    dates=set()
    for _,r in s.iterrows():
        curr = r['Situation'].upper()
        if prev=="GREEN" and curr=="RED": dates.add(r['date'])
        prev = curr
    return ", ".join(str(d) for d in sorted(dates))

def answer_q21(df):
    tmp = df[df["Situation"].str.upper().isin(["GREEN","YELLOW"])]
    cnt = tmp.groupby(tmp['Timestamp'].dt.date).size()
    return ", ".join(d.strftime("%Y-%m-%d") for d in cnt.nlargest(3).index)

def answer_q22(df):
    m   = df['Timestamp'].dt.to_period('M')
    cnt = df[df["Situation"].str.upper()=="GREEN"].groupby(m).size()
    return ", ".join(cnt.nlargest(3).index.astype(str))

def answer_q23(df):
    v = df['Value']
    return f"Genel dalgalanma: {v.min():.2f}–{v.max():.2f} mm/s; ortalama {v.mean():.2f}"

def answer_q24(df):
    ch  = df["Situation"].str.upper().ne(df["Situation"].str.upper().shift())
    top = df[ch].groupby(df['date']).size().nlargest(3).index
    return ", ".join(str(d) for d in top)

def answer_q25(df):
    err = df.groupby('date')['Situation'].apply(lambda s: (s.str.upper()!="GREEN").mean())
    return str(err.idxmin())

def answer_q26(df):
    mon = df.groupby(df['Timestamp'].dt.to_period('M'))['Value'].std()
    return str(mon.idxmax())

def answer_q27(df):
    ann = df.groupby(df['Timestamp'].dt.year)['Value'].mean()
    return ", ".join(f"{y}:{v:.2f}" for y,v in ann.items())

def answer_q28(df):
    trans, prev = 0, df.sort_values("Timestamp").iloc[0]['Situation'].upper()
    for s in df['Situation'].str.upper()[1:]:
        if prev=="GREEN" and s=="RED": trans+=1
        prev=s
    return str(trans)

def answer_q29(df):
    rec = df[df['Timestamp'] >= df['Timestamp'].max()-pd.Timedelta(days=30)]
    return ", ".join(sorted(rec['Situation'].unique()))

def answer_q30(df):
    rc  = df[df["Situation"].str.upper()=="RED"].groupby('date').size()
    top = rc[rc==rc.max()].index
    return ", ".join(str(d) for d in top)

def answer_q31(df):
    return "Düzenli bakım ve sensör kalibrasyonu önerilir."

def answer_q32(df):
    return "Titreşim, fan yatak aşınması ve balans dengesizliği performansı etkileyebilir."

def answer_all_red_dates(df):
    red_dates = sorted(df[df["Situation"].str.upper()=="RED"]["date"].unique())
    return ", ".join(str(d) for d in red_dates)

# ─────────────────────────────────────────────────────────────────────────────
# 4️⃣ Soru–Fonksiyon eşlemesi
qa_map = [
    ("Makinenin en fazla arıza yaptığı tarih aralığını verebilir misin?",       answer_q1),
    ("Geçmişte sarı alarm seviyesinde en çok hangi tarihlerde görüldü?",        answer_q2),
    ("Geçmişte kırmızı alarm seviyesinde en çok hangi tarihlerde görüldü?",     answer_q3),
    ("Makine performansındaki dalgalanmalar hakkında bilgi verebilir misin?",   answer_q4),
    ("2023 aralığında makinede kaç kez alarm durumu oluştu?",                   answer_q5),
    ("Makine hangi değerlerde sarı alarma geçiyor?",                            answer_q6),
    ("Makine hangi değerlerde kırmızı alarma geçiyor?",                         answer_q7),
    ("… tarihinde RTF makinesinin değer aralığı neydi?",                        answer_q8),
    ("… tarihinde RTF makinesinin renk seviyesi neydi?",                       answer_q9),
    ("… tarihinde makine performansı hakkında bilgi alabilir miyim?",           answer_q10),
    ("RTF makinesi 2023 aralığında turuncu alarm seviyesinde çalıştığı tarihler?", answer_q11),
    ("2023 yılında makine performansı nasıldı?",                                answer_q12),
    ("Son bir yıl içinde makine hangi günlerde tamamen durdu?",                 answer_q13),
    ("Kırmızı seviyede çalıştığı tarihlerde makine tamamen durdu mu?",          answer_q14),
    ("15 Ocak 2023’ten itibaren yeşil alarmda çalışılan toplam süre nedir?",    answer_q15),
    ("Son üç ayda sarı alarm seviyesinde kaç gün çalıştı?",                     answer_q16),
    ("Son üç ayda turuncu alarm seviyesinde kaç gün çalıştı?",                  answer_q17),
    ("Son üç ayda yeşil alarm seviyesinde kaç gün çalıştı?",                    answer_q18),
    ("Son üç ayda kırmızı alarm seviyesinde kaç gün çalıştı?",                   answer_q19),
    ("Yeşilden direkt kırmızıya geçiş yapan günler hangileri?",                answer_q20),
    ("Hangisi renk değişimlerinin en düzenli olduğu tarihler?",                 answer_q21),
    ("Makine, son bir yılda hangi aylarda daha çok yeşildi?",                  answer_q22),
    ("Makine performansının zaman içindeki değişimi nasıl oldu?",              answer_q23),
    ("Renk değişimlerinin yoğun olduğu dönemler hangi tarihler?",              answer_q24),
    ("En düşük arıza oranı hangi tarihlerde?",                                answer_q25),
    ("En fazla performans değişikliği hangi ayda oldu?",                       answer_q26),
    ("Her yılın ortalama performansı nedir?",                                  answer_q27),
    ("Yeşilden kırmızıya kaç kez geçiş yapıldı?",                              answer_q28),
    ("Son bir ayda hangi renk aralıklarında çalıştı?",                          answer_q29),
    ("En yüksek değerlerde alarm verdiği günler hangileri?",                   answer_q30),
    ("Makine iyileştirme önerileri nelerdir?",                                 answer_q31),
    ("Makinenin performansını ne etkileyebilir?",                              answer_q32),
    ("Tüm kırmızı günleri listele",                                            answer_all_red_dates)
]

# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣ Embedder + FAISS index
embedder_q = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
questions  = [q for q,_ in qa_map]
Q_emb      = embedder_q.encode(questions, convert_to_numpy=True)
faiss.normalize_L2(Q_emb)
idx_q      = faiss.IndexFlatIP(Q_emb.shape[1])
idx_q.add(Q_emb)

# ─────────────────────────────────────────────────────────────────────────────
# 6️⃣ rag_answer: hem date-parametrik hem LLM fallback
def rag_answer(
    user_q: str,
    df: pd.DataFrame,
    model=None,
    tokenizer=None,
    threshold: float = 0.65,
    date: str | None = None
) -> str:
    # 1) FAISS Retrieval
    ue = embedder_q.encode([user_q], convert_to_numpy=True)
    faiss.normalize_L2(ue)
    Dq, Iq = idx_q.search(ue, 1)

    if Dq[0][0] >= threshold:
        fn  = qa_map[Iq[0][0]][1]
        sig = inspect.signature(fn).parameters
        # Tarih parametreli mi?
        if len(sig) == 2:
            date = date or extract_date(user_q)
            if date is None:
                return "Lütfen sorunuzda bir tarih belirtin (örn. “15 Haziran 2023”)."
            return fn(df, date)
        return fn(df)

    # 2) LLM fallback
    if model and tokenizer:
        prompt = f"Soru: {user_q}\nCevap:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out    = model.generate(**inputs, max_new_tokens=300)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    # 3) Hiçbirinden cevap gelmediyse
    return "Cevap bulunamadı."

