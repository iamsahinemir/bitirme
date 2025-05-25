# batch_predict.py
import os
import pandas as pd
from gradio_client import Client
from concurrent.futures import ThreadPoolExecutor, as_completed

def ask_model(q: str) -> str:
    # Thread içinde kullanılacak client, global olarak main()'de tanımlı
    return client.predict(user_question=q, api_name="/predict")

def main():
    # ——————————— Ayarlar ———————————
    SPACE      = "iamsahinemir/bitirme-model"
    INPUT_CSV  = "Chatbot_Sorular_Varyantlar_.csv"
    OUTPUT_CSV = "Chatbot_Sorular_Varyantlar_Cevapli.csv"

    cwd = os.getcwd()
    print(f"Çalışma dizini: {cwd}")
    print(f"Girdi CSV tam yolu: {os.path.join(cwd, INPUT_CSV)}")

    # ——————————— CSV'yi oku ———————————
    df = pd.read_csv(INPUT_CSV)
    questions = df["Soru"].tolist()
    n = len(questions)
    print(f"Toplam {n} soru yüklendi.")

    # ——————————— Client oluştur ———————————
    global client
    client = Client(SPACE)

    # ——————————— Paralel tahmin et (ThreadPool) ———————————
    answers = [None] * n
    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(ask_model, q): i for i, q in enumerate(questions)}
        for count, fut in enumerate(as_completed(futures), 1):
            idx = futures[fut]
            try:
                answers[idx] = fut.result()
            except Exception as e:
                answers[idx] = f"ERROR: {e}"
            print(f"{count}/{n} tamamlandı")

    # ——————————— Sonuçları kaydet ———————————
    df["Cevap"] = answers
    out_path = os.path.join(cwd, OUTPUT_CSV)
    try:
        df.to_csv(out_path, index=False)
        print(f"✅ Bitti: {out_path}")
    except Exception as e:
        print(f"❌ CSV yazılamadı: {e}")

    if os.path.exists(out_path):
        print("📂 Dosya başarıyla oluşturuldu.")
    else:
        print("⚠️ Dosya bulunamadı!")

if __name__ == "__main__":
    main()
