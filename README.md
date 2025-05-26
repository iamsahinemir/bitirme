---
title: bitirme-model  
emoji: 🧠  
colorFrom: indigo  
colorTo: pink  
sdk: gradio  
sdk_version: 5.29.1  
app_file: app.py  
pinned: false  
---

# Makine Titreşim Asistanı / Machine Vibration Assistant

**Türkçe:** Bu Space, LLaMA + LoRA + RAG kullanarak makine titreşim verisi üzerinden soruları yanıtlayan bir Türkçe asistan içerir.  
**English:** This Space provides a Turkish-language assistant that answers questions over machine vibration data using LLaMA + LoRA + RAG.

---

## 🚀 Özellikler / Features

- **⚙️ Model & RAG**  
  - Meta-LLaMA + LoRA adapter (8-bit) ile Türkçe endüstriyel sohbet  
  - FAISS tabanlı statik QA ve dinamik veri retrieval  
- **🖥️ Arayüz / Interface (Gradio)**  
  - Sohbet geçmişi: yeni, yeniden adlandır, sil / Chat history: new, rename, delete  
  - 🎨 Koyu/Açık tema butonu / Dark ↔ Light theme toggle  
  - 🗓️ Grafiksel analiz: tarih aralığı filtrelemesi, anomali tespiti, günlük olay & kök-neden dağılımı  
    / Date-range filtering, anomaly detection, daily incidents & root-cause charts  
- **📈 Görselleştirme / Visualization**  
  - Scatter plot with anomaly markers  
  - Bar chart of daily incident counts  
  - Pie chart of root cause distribution  
- **🗓️ Tarih Validasyonu / Date Validation**  
  - Sadece 2023-01-08 … 2023-12-23 arası / Only valid between 2023-01-08 and 2023-12-23  
  - Hatalı format veya aralıkta uyarı mesajı / Error if invalid format or out of range  

---

## 📥 Kurulum ve Çalıştırma / Installation & Usage

```bash
git clone https://github.com/iamsahinemir/bitirme.git
cd bitirme
pip install -r requirements.txt
python app.py
```

- **Türkçe:** `vibration_df.csv` dosyasını proje köküne koymayı unutmayın.  
- **English:** Place the `vibration_df.csv` file in the project root.  
- **Public Link:** Gradio’da `demo.launch(share=True)` ile paylaşılabilir.

---

## 📄 Lisans / License

**MIT License**  
© 2025 Emir Esad Şahin  

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

**Türkçe:** Bu proje **MIT Lisansı** ile lisanslanmıştır.  
**English:** This project is licensed under the **MIT License**.

---

## 🤝 Katkıda Bulunun / Contributing

- **Türkçe:** Issue ve PR’lere açığız. Kod stilini Black ile formatlayın; test ekleyin.  
- **English:** Feel free to open issues & PRs. Please format code with Black and include tests.  
- **Contact / İletişim:** iamsahinemir@gmail.com 

---
