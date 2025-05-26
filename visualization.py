# visualization.py
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# --- Sabit metin ve veriler ---
EQUIP_TEXT = (
    "Ekipmanın İşlevi:\n"
    "Galvaniz hatlarındaki RTF (Radiant Tube Furnace) Fırın Yakma Havası Fanı, "
    "fırında yanma için gerekli oksijeni sağlayarak yakıtın verimli bir şekilde yanmasını mümkün kılar. "
    "Doğru çalışmaması durumunda düzensiz yanma, sıcaklık dalgalanmaları ve enerji kayıpları gibi sorunlar ortaya çıkabilir."
)

EXPLANATIONS = {
    'Process based alert': "Yanlış sensör kalibrasyonu veya ani değişimler. Süreçte durma, bakım artışı.",
    'Bearing fault': "Yağsızlık, kir, hizasızlık. Titreşim artışı, rulman hasarı.",
    'Balance fault': "Rotor dengesizliği. Sallanma, yorulma, rezonans."
}

dates = pd.to_datetime([
    '2023-02-01','2023-04-26','2023-04-27','2023-04-28','2023-04-29',
    '2023-04-30','2023-05-02','2023-05-03','2023-07-03','2023-10-04',
    '2023-10-05','2023-10-06','2023-12-10']
)
incidents_df = pd.DataFrame({
    'Date': dates,
    'Incidents': [1,25,78,42,29,23,1,4,1,2,3,15,1],
    'Root Cause': [
        'Process based alert','Bearing fault','Bearing fault','Bearing fault','Bearing fault','Bearing fault',
        'Bearing fault','Bearing fault','Process based alert','Balance fault','Balance fault','Balance fault','Process based alert'
    ]
}).set_index('Date')

# --- Fonksiyonlar ---
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
    df = load_and_process_data()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = df[(df['Timestamp'] >= start) & (df['Timestamp'] <= end)].copy()
    if filtered.empty:
        return "Seçilen tarih aralığında veri bulunamadı.", None

    filtered['Color'] = filtered['Value'].apply(get_color)
    model = IsolationForest(contamination=0.01, random_state=42)
    filtered['Anomaly'] = model.fit_predict(filtered[['Value']])
    filtered['AnomalyColor'] = filtered['Anomaly'].map({1: 'blue', -1: 'red'})

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    axs[0].scatter(filtered['Timestamp'], filtered['Value'], c=filtered['AnomalyColor'])
    axs[0].set_title("Anomaly Detection - Velocity")
    axs[0].set_ylabel("Velocity (mm/s)")

    axs[1].bar(incidents_df.index, incidents_df['Incidents'], width=1)
    axs[1].set_title("Daily Incidents")

    root_counts = incidents_df['Root Cause'].value_counts()
    axs[2].pie(root_counts.values, labels=root_counts.index, autopct='%1.1f%%')
    axs[2].set_title("Root Cause Distribution")

    return "Grafikler oluşturuldu.", fig

# --- Arayüz ---
with gr.Blocks() as demo:
    gr.Markdown("## 🔧 RTF Titreşim Anomali Görselleştirme")
    gr.Image("rtfmachine.gif", elem_id="rtf-gif", show_label=False, show_download_button=False)
    gr.Markdown(EQUIP_TEXT)

    with gr.Row():
        start_date = gr.Textbox(label="Başlangıç Tarihi (YYYY-MM-DD)", placeholder="2023-01-01")
        end_date = gr.Textbox(label="Bitiş Tarihi (YYYY-MM-DD)", placeholder="2023-12-31")

    plot_btn = gr.Button("🔍 Veriyi Göster ve Anomali Algıla")
    plot_output = gr.Textbox(label="Durum")
    chart_output = gr.Plot()

    plot_btn.click(fn=plot_graphs, inputs=[start_date, end_date], outputs=[plot_output, chart_output])

if __name__ == "__main__":
    demo.launch(server_port=7861)
