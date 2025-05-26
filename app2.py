import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime
import json
import os

chat_histories = {}
MAX_BUTTONS = 30

# ------------------------------ Chat Fonksiyonları ------------------------------
def save_chat_histories():
    with open("chat_histories.json", "w", encoding="utf-8") as f:
        json.dump(chat_histories, f, ensure_ascii=False, indent=2)

def load_chat_histories():
    global chat_histories
    if os.path.exists("chat_histories.json"):
        with open("chat_histories.json", "r", encoding="utf-8") as f:
            chat_histories.update(json.load(f))

def generate_answer(message):
    return f"🧠 (Mock Cevap): '{message}' sorusu alındı."

def submit_message(message, history_id):
    if not history_id:
        history_id = f"Sohbet 1"
    if history_id not in chat_histories:
        chat_histories[history_id] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    answer = generate_answer(message)
    chat_histories[history_id].append({"role": "user", "content": f"{timestamp}\n🧑 {message}"})
    chat_histories[history_id].append({"role": "assistant", "content": f"{timestamp}\n🤖 {answer}"})
    save_chat_histories()
    sohbetler = list(chat_histories.keys())
    return chat_histories[history_id], "", sohbetler, history_id

def new_chat():
    new_id = f"Sohbet {len(chat_histories)+1}"
    chat_histories[new_id] = []
    save_chat_histories()
    sohbetler = list(chat_histories.keys())
    return sohbetler, [], "", new_id

def load_chat(selected_id):
    return chat_histories.get(selected_id, []), "", selected_id, selected_id

def temizle_sohbet(history_id):
    if history_id in chat_histories:
        chat_histories[history_id] = []
        save_chat_histories()
    return [], "", history_id

def sil_sohbet(sil_id):
    mesaj = ""
    if sil_id in chat_histories:
        del chat_histories[sil_id]
        save_chat_histories()
        mesaj = f"🗑️ '{sil_id}' başarıyla silindi."
    sohbetler = list(chat_histories.keys())
    return sohbetler, [], "", mesaj

def yeniden_adlandir(onceki_id, yeni_id):
    if onceki_id in chat_histories and yeni_id:
        chat_histories[yeni_id] = chat_histories.pop(onceki_id)
        save_chat_histories()
    sohbetler = list(chat_histories.keys())
    return sohbetler, [], yeni_id

def update_sidebar(sohbet_ids):
    return [
        gr.update(visible=True, value=sohbet_ids[i]) if i < len(sohbet_ids)
        else gr.update(visible=False)
        for i in range(MAX_BUTTONS)
    ]

# ------------------------------ Grafik Fonksiyonları ------------------------------
def load_and_process_data():
    df = pd.read_csv("vibration_df.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(df['Timestamp'].dtype):
        df['Timestamp'] = df['Timestamp'].dt.tz_convert(None)
    df.dropna(subset=['Timestamp'], inplace=True)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.dropna(subset=['Value'], inplace=True)
    df.sort_values('Timestamp', inplace=True)
    return df

def get_color(v):
    if v < 2.8: return 'green'
    if v < 11.2: return 'yellow'
    if v < 14: return 'orange'
    return 'red'

def plot_graphs(start_date, end_date):
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "❌ Lütfen tarihi 'YYYY-MM-DD' formatında giriniz.", None

    # Geçerli veri aralığı
    min_date = datetime(2023, 1, 8)
    max_date = datetime(2023, 12, 23)

    if start > end:
        return "❌ Başlangıç tarihi, bitiş tarihinden sonra olamaz.", None
    if start < min_date or end > max_date:
        return f"❌ Tarih aralığı {min_date.date()} ile {max_date.date()} arasında olmalıdır.", None

    df = load_and_process_data()
    end += pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = df[(df['Timestamp'] >= start) & (df['Timestamp'] <= end)].copy()

    if filtered.empty:
        return "⚠️ Seçilen tarih aralığında veri bulunamadı.", None

    filtered['Color'] = filtered['Value'].apply(get_color)
    model = IsolationForest(contamination=0.01, random_state=42)
    filtered['Anomaly'] = model.fit_predict(filtered[['Value']])
    filtered['AnomalyColor'] = filtered['Anomaly'].map({1: 'blue', -1: 'red'})

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(filtered['Timestamp'], filtered['Value'], c=filtered['AnomalyColor'])
    ax.set_title("Anomali Tespiti - Hız")
    ax.set_ylabel("mm/s")
    ax.set_xlabel("Zaman")

    return "✅ Grafik oluşturuldu.", fig


# ------------------------------ Arayüz ------------------------------
load_chat_histories()

with gr.Blocks(css="""
.sidebar-btn { margin-bottom: 4px; border-radius: 6px; text-align: left; padding: 8px; background-color: #1e1e1e; color: white; border: 1px solid #444; }
.sidebar-btn:hover { background-color: #333; cursor: pointer; }
.rename-panel { display: flex; gap: 5px; margin-top: 10px; }
.alert-danger {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
    padding: 10px;
    border-radius: 6px;
    margin-top: 10px;
}

""") as demo:

    with gr.Tabs():
        with gr.Tab("💬 Sohbet"):
            gr.Markdown("## 🤖 Makine Titreşim Asistanı")

            with gr.Row():
                with gr.Column(scale=2):
                    yeni_btn = gr.Button("➕ Yeni Sohbet")
                    tema_html = gr.HTML("""<button onclick="
                        const b = document.body;
                        b.classList.toggle('dark');
                        b.classList.toggle('light');
                    " style='padding:8px;border-radius:6px;background:#222;color:white;border:1px solid #444;'>🌙 Tema Değiştir</button>""")

                    sohbet_buttons = [gr.Button(visible=False, elem_classes="sidebar-btn") for _ in range(MAX_BUTTONS)]

                with gr.Column(scale=8):
                    sohbet_ekrani = gr.Chatbot(label="Sohbet", show_copy_button=True, type="messages")
                    mesaj_kutusu = gr.Textbox(placeholder="Bir mesaj yazın...", show_label=False)
                    gonder_btn = gr.Button("Gönder")
                    temizle_btn = gr.Button("Temizle")

                    gr.Markdown("### Seçilen Sohbet İşlemleri")
                    alert_kutusu = gr.Markdown("", elem_classes="alert-danger", visible=False)
                    with gr.Row(elem_classes="rename-panel"):
                        ad_input = gr.Textbox(label="Yeni Ad", scale=2)
                        ad_btn = gr.Button("✏️ Adı Güncelle", scale=1)
                        sil_btn = gr.Button("🔚 Sil", scale=1)

                    secili_sohbet = gr.State("")
                    sohbet_id_listesi = gr.State(list(chat_histories.keys()))

                    demo.load(fn=lambda: update_sidebar(list(chat_histories.keys())), outputs=sohbet_buttons)

                    yeni_btn.click(fn=new_chat,
                                   outputs=[sohbet_id_listesi, sohbet_ekrani, mesaj_kutusu, secili_sohbet])\
                        .then(fn=update_sidebar, inputs=[sohbet_id_listesi], outputs=sohbet_buttons)

                    gonder_btn.click(fn=submit_message,
                                     inputs=[mesaj_kutusu, secili_sohbet],
                                     outputs=[sohbet_ekrani, mesaj_kutusu, sohbet_id_listesi, secili_sohbet])\
                        .then(fn=update_sidebar, inputs=[sohbet_id_listesi], outputs=sohbet_buttons)

                    mesaj_kutusu.submit(fn=submit_message,
                                        inputs=[mesaj_kutusu, secili_sohbet],
                                        outputs=[sohbet_ekrani, mesaj_kutusu, sohbet_id_listesi, secili_sohbet])\
                        .then(fn=update_sidebar, inputs=[sohbet_id_listesi], outputs=sohbet_buttons)

                    temizle_btn.click(fn=temizle_sohbet,
                                      inputs=[secili_sohbet],
                                      outputs=[sohbet_ekrani, mesaj_kutusu, secili_sohbet])

                    sil_btn.click(fn=sil_sohbet,
              inputs=[secili_sohbet],
              outputs=[sohbet_id_listesi, sohbet_ekrani, secili_sohbet, alert_kutusu])\
    .then(fn=update_sidebar, inputs=[sohbet_id_listesi], outputs=sohbet_buttons)\
    .then(lambda: gr.update(visible=True), outputs=alert_kutusu)\
    .then(lambda: "", outputs=ad_input)  # 🛠️ Ad input temizlensin


                    ad_btn.click(fn=yeniden_adlandir,
                                 inputs=[secili_sohbet, ad_input],
                                 outputs=[sohbet_id_listesi, sohbet_ekrani, secili_sohbet])\
                        .then(fn=update_sidebar, inputs=[sohbet_id_listesi], outputs=sohbet_buttons)

                    for btn in sohbet_buttons:
                        btn.click(fn=load_chat,
                                  inputs=[btn],
                                  outputs=[sohbet_ekrani, mesaj_kutusu, secili_sohbet, ad_input])

        with gr.Tab("📊 Anomali Görselleştirme"):
            gr.Markdown("### 📈 Titreşim Anomali Grafiği ve Durumlar")
            gr.Image("rtfmachine.gif", width=300)
            with gr.Row():
                start_date = gr.Textbox(label="Başlangıç Tarihi", placeholder="2023-01-01")
                end_date = gr.Textbox(label="Bitiş Tarihi", placeholder="2023-12-31")
            plot_btn = gr.Button("🔍 Anomali Göster")
            durum = gr.Textbox(label="Durum")
            grafik = gr.Plot()
            plot_btn.click(fn=plot_graphs, inputs=[start_date, end_date], outputs=[durum, grafik])

demo.launch()
