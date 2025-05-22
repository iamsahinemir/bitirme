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
df['Timestamp'] = df['Timestamp'].apply(date_parser.isoparse)
if pd.api.types.is_datetime64tz_dtype(df['Timestamp'].dtype):
    df['Timestamp'] = df['Timestamp'].dt.tz_convert(None)
df['date'] = df['Timestamp'].dt.date

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣ Tarih normalizasyonu / çıkarma
turkish_months = {
    "ocak":"January","şubat":"February","mart":"March","nisan":"April",
    "mayıs":"May","haziran":"June","temmuz":"July","ağustos":"August",
    "eylül":"September","ekim":"October","kasım":"November","aralık":"December"
}

def normalize_date_str(s: str) -> str | None:
    s_clean = s.strip().lower()
    s_clean = re.sub(r"[\/\.]", "-", s_clean)
    s_clean = s_clean.replace(",", "")
    for tr, en in turkish_months.items():
        if tr in s_clean:
            s_clean = s_clean.replace(tr, en.lower())
            break
    dt = pd.to_datetime(s_clean, dayfirst=True, errors="coerce")
    return dt.date().isoformat() if not pd.isna(dt) else None

def extract_date(user_q: str) -> str | None:
    # YYYY-MM-DD veya DD-MM-YYYY
    m = re.search(r"(\d{4}[-\/\.]\d{2}[-\/\.]\d{2})|(\d{2}[-\/\.]\d{2}[-\/\.]\d{4})", user_q)
    if m: return normalize_date_str(m.group(0))
    # Türkçe ay isimli
    m_tr = re.search(
        r"(\d{1,2})\s+(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+(\d{4})",
        user_q, flags=re.IGNORECASE
    )
    if m_tr: return normalize_date_str(m_tr.group(0))
    # İngilizce ay isimli
    m_en = re.search(
        r"(\d{1,2}\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2},\s*)?(\d{4})",
        user_q, flags=re.IGNORECASE
    )
    if m_en: return normalize_date_str(m_en.group(0))
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣ 32 QA fonksiyonları (kullanıcı dostu açıklamalarla)

def answer_q1(df):
    red = df[df["Situation"].str.upper()=="RED"].sort_values("Timestamp")
    segs, start, prev = [], red.iloc[0]["Timestamp"], red.iloc[0]["Timestamp"]
    for t in red["Timestamp"].iloc[1:]:
        if (t - prev) > pd.Timedelta(minutes=1):
            segs.append((start, prev)); start = t
        prev = t
    segs.append((start, prev))
    s,e = max(segs, key=lambda x: x[1]-x[0])
    return (
        f"Makinenin arızalı (kırmızı alarm) durumda en uzun süre çalıştığı zaman aralığı "
        f"{s.strftime('%Y-%m-%d %H:%M')} ile {e.strftime('%Y-%m-%d %H:%M')} arasındadır."
    )

def answer_q2(df):
    cnt = df[df["Situation"].str.upper()=="YELLOW"].groupby("date").size()
    dates = ", ".join(str(d) for d in cnt[cnt==cnt.max()].index)
    return (
        f"Makine sarı alarm seviyesinde en sık {dates} tarihlerinde çalışmıştır. "
        "Bu tarihlerde sistemde orta düzeyli uyarılar gözlemlenmiştir."
    )

def answer_q3(df):
    cnt = df[df["Situation"].str.upper()=="RED"].groupby("date").size()
    dates = ", ".join(str(d) for d in cnt[cnt==cnt.max()].index)
    return (
        f"Kırmızı alarm (yüksek titreşim) en çok şu tarihlerde gözlemlenmiştir: {dates}. "
        "Bu tarihlerde arıza riskleri yüksekti."
    )

def answer_q4(df):
    vmin, vmax = df["Value"].min(), df["Value"].max()
    return (
        f"Makine titreşim değerleri {vmin:.2f} mm/s ile {vmax:.2f} mm/s arasında dalgalanmıştır. "
        "Bu aralık, sistem performansının zaman içindeki değişkenliğini gösterir."
    )

def answer_q5(df):
    mask = (df["Timestamp"] >= "2023-01-01") & (df["Timestamp"] <= "2023-12-31")
    count = df[mask & (df["Situation"].str.upper()=="RED")].shape[0]
    return (
        f"2023 yılı boyunca makine toplam {count} kez kırmızı alarm vermiştir. "
        "Bu sayı, yüksek titreşimli arıza olaylarını temsil eder."
    )

def answer_q6(df):
    return (
        "Makine titreşim seviyelerine göre alarm eşikleri şu şekildedir:\n"
        "🟢 Green: 0–2.8 mm/s\n"
        "🟡 Yellow: 2.8–11.2 mm/s\n"
        "🟠 Orange: 11.2–14 mm/s\n"
        "🔴 Red: 14+ mm/s\n"
        "Bu değerler, farklı alarm seviyelerinin sınırlarını belirtir."
    )

def answer_q7(df):
    return (
        "Makine titreşim değeri 14 mm/s’yi aştığında kırmızı alarm seviyesine geçer. "
        "Bu, ciddi bir arıza veya dengesizlik belirtisidir."
    )

def answer_q8(df, date):
    sub = df[df['date']==pd.to_datetime(date).date()]
    return (
        f"{date} tarihinde makinenin titreşim değer aralığı "
        f"{sub['Value'].min():.2f}–{sub['Value'].max():.2f} mm/s olarak gözlemlenmiştir. "
        "Bu aralık o günkü operasyonel koşulları yansıtır."
    )

def answer_q9(df, date):
    sub = df[df['date']==pd.to_datetime(date).date()]
    mode = sub['Situation'].mode()[0]
    return (
        f"{date} tarihinde makinenin baskın alarm seviyesi: {mode}. "
        "Bu, gün boyunca en sık gözlemlenen durumdur."
    )

def answer_q10(df, date):
    sub = df[df['date']==pd.to_datetime(date).date()]
    mn,mx,m = sub['Value'].min(), sub['Value'].max(), sub['Value'].mean()
    return (
        f"{date} tarihinde makinenin minimum titreşimi {mn:.2f} mm/s, "
        f"maksimumu {mx:.2f} mm/s ve ortalaması {m:.2f} mm/s olarak ölçülmüştür."
    )

def answer_q11(df):
    mask = (df['date']>=pd.to_datetime("2023-01-01").date()) & (
           df['date']<=pd.to_datetime("2023-12-31").date())
    dates = df[mask & (df["Situation"].str.upper()=="ORANGE")].groupby("date").size()
    list_dates = ", ".join(str(d) for d in dates[dates>0].index)
    return (
        f"2023 yılında makine aşağıdaki tarihlerde turuncu alarm seviyesinde çalışmıştır: "
        f"{list_dates}."
    )

def answer_q12(df):
    vals = df[df['Timestamp'].dt.year==2023]['Value']
    return (
        f"2023 yılı performansı: minimum {vals.min():.2f} mm/s, maksimum {vals.max():.2f} mm/s, "
        f"ortalama {vals.mean():.2f} mm/s."
    )

def answer_q13(df):
    all_dates = pd.date_range(df['date'].min(), df['date'].max()).date
    missing   = sorted(set(all_dates) - set(df['date'].unique()))
    return (
        "Makine verilerinde ölçüm yapılmayan veya eksik veri olan günler: "
        f"{', '.join(str(d) for d in missing)}."
    )

def answer_q14(df):
    total = df.groupby('date').size()
    red   = df[df['Situation'].str.upper()=='RED']\
             .groupby('date').size()\
             .reindex(total.index, fill_value=0)
    full  = red[ red == total ].index
    return (
        "Makinenin yalnızca kırmızı alarm seviyesinde çalıştığı günler: "
        f"{', '.join(str(d) for d in full)}."
    )

def answer_q15(df):
    mins = df[df['Situation'].str.upper()=="GREEN"].shape[0]
    hours = mins/60
    return (
        f"Yeşil alarm durumunda toplam çalışma süresi: {hours:.2f} saat "
        f"({mins} dakika). Bu süre makinenin stabil çalıştığı zamanları gösterir."
    )

def answer_q16(df):
    recent = df[df['Timestamp'] >= df['Timestamp'].max() - pd.Timedelta(days=90)]
    days = recent[recent["Situation"].str.upper()=="YELLOW"]['date'].nunique()
    return f"Son 3 ayda makine toplam {days} farklı günde sarı alarm seviyesinde çalışmıştır."

def answer_q17(df):
    recent = df[df['Timestamp'] >= df['Timestamp'].max() - pd.Timedelta(days=90)]
    days = recent[recent["Situation"].str.upper()=="ORANGE"]['date'].nunique()
    return f"Son 3 ayda makine toplam {days} farklı günde turuncu alarm seviyesinde çalışmıştır."

def answer_q18(df):
    recent = df[df['Timestamp'] >= df['Timestamp'].max() - pd.Timedelta(days=90)]
    days = recent[recent["Situation"].str.upper()=="GREEN"]['date'].nunique()
    return f"Son 3 ayda makine toplam {days} farklı günde yeşil alarm seviyesinde çalışmıştır."

def answer_q19(df):
    recent = df[df['Timestamp'] >= df['Timestamp'].max() - pd.Timedelta(days=90)]
    days = recent[recent["Situation"].str.upper()=="RED"]['date'].nunique()
    return f"Son 3 ayda makine toplam {days} farklı günde kırmızı alarm seviyesinde çalışmıştır."

def answer_q20(df):
    s = df.sort_values("Timestamp")
    prev = s.iloc[0]['Situation'].upper()
    dates = set()
    for _,r in s.iterrows():
        curr = r['Situation'].upper()
        if prev=="GREEN" and curr=="RED":
            dates.add(r['date'])
        prev = curr
    return (
        "Makine doğrudan yeşil alarmdan kırmızı alarma geçtiği tarihler: "
        f"{', '.join(str(d) for d in sorted(dates))}."
    )

def answer_q21(df):
    tmp = df[df["Situation"].str.upper().isin(["GREEN","YELLOW"])]
    cnt = tmp.groupby(tmp['Timestamp'].dt.date).size()
    dates = cnt.nlargest(3).index.strftime("%Y-%m-%d")
    return (
        "En düzenli renk değişimleri gözlemlenen tarihler: "
        f"{', '.join(dates)}."
    )

def answer_q22(df):
    m   = df['Timestamp'].dt.to_period('M')
    cnt = df[df["Situation"].str.upper()=="GREEN"].groupby(m).size()
    months = cnt.nlargest(3).index.astype(str)
    return (
        "Son bir yılda makinenin en fazla yeşil alarmda olduğu aylar: "
        f"{', '.join(months)}."
    )

def answer_q23(df):
    vmin, vmax, vmean = df['Value'].min(), df['Value'].max(), df['Value'].mean()
    return (
        f"Genel titreşim performansı: minimum {vmin:.2f} mm/s, maksimum {vmax:.2f} mm/s, "
        f"ortalama {vmean:.2f} mm/s. Bu değerler sistem stabilitesini yansıtır."
    )

def answer_q24(df):
    ch  = df["Situation"].str.upper().ne(df["Situation"].str.upper().shift())
    top = df[ch].groupby(df['date']).size().nlargest(3).index
    return (
        "Renk değişimlerinin yoğun olduğu tarihler: "
        f"{', '.join(str(d) for d in top)}."
    )

def answer_q25(df):
    err = df.groupby('date')['Situation']\
            .apply(lambda s: (s.str.upper()!="GREEN").mean())
    date = err.idxmin()
    return (
        f"Makinenin en az arıza yaptığı gün: {date}. Bu gün büyük oranda yeşil alarm seviyesinde kalmıştır."
    )

def answer_q26(df):
    mon = df.groupby(df['Timestamp'].dt.to_period('M'))['Value'].std()
    month = mon.idxmax()
    return (
        f"Makine en fazla performans değişikliğini {month} ayında yaşamıştır. "
        "Bu ayda titreşim değerlerindeki dalgalanma en yüksekti."
    )

def answer_q27(df):
    ann = df.groupby(df['Timestamp'].dt.year)['Value'].mean()
    summary = ", ".join(f"{y}: {v:.2f}" for y,v in ann.items())
    return f"Yıllık ortalama titreşim değerleri: {summary} mm/s."

def answer_q28(df):
    trans, prev = 0, df.sort_values("Timestamp").iloc[0]['Situation'].upper()
    for s in df['Situation'].str.upper()[1:]:
        if prev=="GREEN" and s=="RED":
            trans += 1
        prev = s
    return (
        f"Yeşil alarmdan kırmızı alarma toplam {trans} kez geçiş olmuştur. "
        "Bu geçişler ani bozulma durumlarını gösterir."
    )

def answer_q29(df):
    rec = df[df['Timestamp'] >= df['Timestamp'].max()-pd.Timedelta(days=30)]
    statuses = ", ".join(sorted(rec['Situation'].unique()))
    return (
        "Son bir ayda gözlemlenen alarm seviyeleri: "
        f"{statuses}."
    )

def answer_q30(df):
    rc  = df[df["Situation"].str.upper()=="RED"].groupby('date').size()
    top = rc[rc==rc.max()].index
    return (
        "En yüksek alarm seviyeleri şu tarihlerde gözlemlenmiştir: "
        f"{', '.join(str(d) for d in top)}."
    )

def answer_q31(df):
    return (
        "Makinenin performansını artırmak için düzenli bakım önerilir, "
        "sensör kalibrasyonu yapılmalı ve çalışma ortamı koşulları izlenmelidir."
    )

def answer_q32(df):
    return (
        "Titreşim performansını etkileyebilecek başlıca faktörler:\n"
        "- Fan yataklarının aşınması\n"
        "- Balans bozulmaları\n"
        "- Mekanik gevşemeler veya darbe etkileri\n"
        "Bu unsurlar düzenli kontrol edilmelidir."
    )

def answer_all_red_dates(df):
    red_dates = sorted(df[df["Situation"].str.upper()=="RED"]["date"].unique())
    return (
        "Makine aşağıdaki tarihlerde kırmızı alarm seviyesindeydi: "
        f"{', '.join(str(d) for d in red_dates)}."
    )

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
# 5️⃣ Embedder + FAISS index (soru tabanlı)
embedder_q = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
questions  = [q for q,_ in qa_map]
Q_emb      = embedder_q.encode(questions, convert_to_numpy=True)
faiss.normalize_L2(Q_emb)
idx_q      = faiss.IndexFlatIP(Q_emb.shape[1])
idx_q.add(Q_emb)

# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣➕ FAISS index (satır tabanlı) — dinamik konteks için
row_texts = df.apply(
    lambda r: f"Tarih: {r['date']}, Durum: {r['Situation']}, Değer: {r['Value']:.2f}",
    axis=1
).tolist()
row_embs = embedder_q.encode(row_texts, convert_to_numpy=True)
faiss.normalize_L2(row_embs)
row_idx  = faiss.IndexFlatIP(row_embs.shape[1])
row_idx.add(row_embs)

# ─────────────────────────────────────────────────────────────────────────────
# 6️⃣ rag_answer: dinamik "son x ay" + öneri + LLM fallback + dynamic context
def rag_answer(
    user_q: str,
    df: pd.DataFrame,
    model=None,
    tokenizer=None,
    threshold: float = 0.65,
    date: str | None = None
) -> str:
    # ———————————————————————————————
    # (0a) Dinamik "Son x ay"
    turkish_numbers = {
        "bir":1, "iki":2, "üç":3, "dört":4, "beş":5,
        "altı":6, "yedi":7, "sekiz":8, "dokuz":9, "on":10
    }
    m_num  = re.search(r"son\s+(\d+)\s+ay", user_q, flags=re.IGNORECASE)
    m_word = re.search(
        rf"son\s+({'|'.join(turkish_numbers.keys())})\s+ay",
        user_q, flags=re.IGNORECASE
    )
    if m_num or m_word:
        x      = int(m_num.group(1)) if m_num else turkish_numbers[m_word.group(1).lower()]
        now    = df['Timestamp'].max()
        cutoff = now - pd.DateOffset(months=x)
        lower  = user_q.lower()
        if "sarı"   in lower:
            color, label = "YELLOW", "sarı alarm seviyesinde"
        elif "turuncu" in lower:
            color, label = "ORANGE", "turuncu alarm seviyesinde"
        elif "yeşil"  in lower:
            color, label = "GREEN",  "yeşil alarm seviyesinde"
        elif "kırmızı" in lower:
            color, label = "RED",    "kırmızı alarm seviyesinde"
        else:
            days = df[df['Timestamp']>=cutoff]['date'].nunique()
            return f"Son {x} ayda toplam {days} farklı gün veri kaydı var."
        days = df[
            (df['Timestamp']>=cutoff) &
            (df['Situation'].str.upper()==color)
        ]['date'].nunique()
        return f"Son {x} ayda makine toplam {days} farklı günde {label} çalışmıştır."

    # ———————————————————————————————
    # (0b) Özel "ayında arıza"
    ay_ariza = re.search(
        r"(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+ayında.*arıza",
        user_q, flags=re.IGNORECASE
    )
    if ay_ariza:
        mon = {
            "ocak":1,"şubat":2,"mart":3,"nisan":4,"mayıs":5,"haziran":6,
            "temmuz":7,"ağustos":8,"eylül":9,"ekim":10,"kasım":11,"aralık":12
        }[ay_ariza.group(1).lower()]
        red_days = df[
            (df['Timestamp'].dt.month==mon) &
            (df['Situation'].str.upper()=="RED")
        ]['date'].unique()
        return (
            f"Evet, şu tarihlerde: {', '.join(map(str,sorted(red_days)))}"
            if len(red_days)
            else "Hayır, o ayda kırmızı (arıza) durumu görülmemiş."
        )

    # ———————————————————————————————
    # (1) Normalize: makine → RTF makinesi
    q = re.sub(r"\bmakinenin\b","RTF makinesinin",user_q,flags=re.IGNORECASE)
    q = re.sub(r"\bmakine\b","RTF makinesi",q,flags=re.IGNORECASE)

    # ———————————————————————————————
    # (2) FAISS retrieval — soru tabanlı
    ue = embedder_q.encode([user_q],convert_to_numpy=True)
    faiss.normalize_L2(ue)
    Dq,Iq = idx_q.search(ue,3)

    # (2a) Direkt eşleşme
    for r in range(3):
        if Dq[0][r] >= threshold:
            fn  = qa_map[Iq[0][r]][1]
            sig = inspect.signature(fn).parameters
            if len(sig) == 2:
                date = date or extract_date(user_q)
                if date is None:
                    return 'Lütfen sorunuzda bir tarih belirtin (örn. "15 Şubat 2023").'
                return fn(df, date)
            return fn(df)

    # (2b) Öneri sistemi
    suggestions = [
        qa_map[Iq[0][r]][0]
        for r in range(3)
        if Dq[0][r] >= 0.4
    ]
    if suggestions:
        return (
            "Tam olarak anlayamadım. Şunları mı demek istediniz?\n\n"
            + "\n".join(f"- {s}" for s in suggestions)
        )

    # ———————————————————————————————
    # (3) LLM fallback + dynamic context
    if model and tokenizer:
        # a) satır tabanlı indeksten top5 al
        D_rows, I_rows = row_idx.search(ue, 5)
        context_rows  = "\n".join(row_texts[i] for i in I_rows[0])
        prompt = (
            "Aşağıda RTF makinesi titreşim verileri var:\n"
            f"{context_rows}\n\n"
            f"Soru: {q}\n"
            "Bu verilere dayanarak cevap verin:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out    = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    return "Cevap bulunamadı."
