import pandas as pd

def main():
    input_csv = "Chatbot_Sorular_Varyantlar_Cevapli.csv"
    output_csv = "Chatbot_Sorular_Varyantlar_DoğruCevaplu.csv"

    # 1) CSV'yi oku
    df = pd.read_csv(input_csv)

    # 2) Doğru Cevap sütunu için boş bir liste oluştur
    dogru_list = [""] * len(df)

    # 3) 5'erli gruplar halinde işleme al
    for start in range(0, len(df), 5):
        main_answer = df.loc[start, "Cevap"]
        # Kalan 4 satırı karşılaştır
        for i in range(start + 1, min(start + 5, len(df))):
            if df.loc[i, "Cevap"] == main_answer:
                dogru_list[i] = "evet"
            else:
                dogru_list[i] = "hayır"

    # 4) Yeni sütunu ekle ve kaydet
    df["Doğru Cevap"] = dogru_list
    df.to_csv(output_csv, index=False)
    print(f"✅ Doğrulama tamamlandı: {output_csv}")

if __name__ == "__main__":
    main()