# batch_predict.py
import pandas as pd
from gradio_client import Client
from concurrent.futures import ProcessPoolExecutor, as_completed

# 1) Ayarlar
SPACE = "iamsahinemir/bitirme-model"          # public Space
INPUT_CSV  = "Chatbot_Sorular_Varyantlar_.csv"
OUTPUT_CSV = "Chatbot_Sorular_Varyantlar_Cevapli.csv"

# 2) CSV'yi oku
df = pd.read_csv(INPUT_CSV)
questions = df["Soru"].tolist()
n = len(questions)

# 3) Tek bir client örneği yaratalım
client = Client(SPACE)

# 4) Soru sorup str cevap döndüren fonksiyon
def ask_model(q: str) -> str:
    try:
        # Burada artık /predict endpoint’i ve user_question parametresi kullanılıyor
        return client.predict(
            user_question=q,
            api_name="/predict"
        )
    except Exception as e:
        return f"ERROR: {e}"

if __name__ == "__main__":
    answers = [None] * n

    # 5) Paralel olarak tüm soruları işleyelim
    with ProcessPoolExecutor() as exe:
        futures = {exe.submit(ask_model, q): i for i, q in enumerate(questions)}
        for count, fut in enumerate(as_completed(futures), 1):
            idx = futures[fut]
            answers[idx] = fut.result()
            print(f"{count}/{n} tamamlandı")

    # 6) Cevapları DataFrame’e ekle ve kaydet
    df["Cevap"] = answers
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Bitti: {OUTPUT_CSV}")
