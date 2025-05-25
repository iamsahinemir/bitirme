import gradio as gr
from datetime import datetime
import json
import os

chat_histories = {}
MAX_BUTTONS = 30

# JSON dosyasına kayıt ve yükleme
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
    # Update the rename input field with current chat name
    return chat_histories.get(selected_id, []), "", selected_id, selected_id

def temizle_sohbet(history_id):
    if history_id in chat_histories:
        chat_histories[history_id] = []
        save_chat_histories()
    return [], "", history_id

def sil_sohbet(sil_id):
    if sil_id in chat_histories:
        del chat_histories[sil_id]
        save_chat_histories()
        # Force save and reload to persist changes
        load_chat_histories()  
    sohbetler = list(chat_histories.keys())
    return sohbetler, [], ""


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

load_chat_histories()

demo = gr.Blocks(
    css="""
body.dark {
    background-color: #0f0f0f !important;
    color: white !important;
}
body.light {
    background-color: white !important;
    color: black !important;
}
#sidebar-scroll {
    max-height: 520px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #555 #1e1e1e;
}
#sidebar-scroll::-webkit-scrollbar {
    width: 8px;
}
#sidebar-scroll::-webkit-scrollbar-track {
    background: #1e1e1e;
}
#sidebar-scroll::-webkit-scrollbar-thumb {
    background-color: #555;
    border-radius: 6px;
    border: 2px solid #1e1e1e;
}
.sidebar-btn {
    margin-bottom: 4px;
    border-radius: 6px;
    text-align: left;
    padding: 8px;
    background-color: #1e1e1e;
    color: white;
    border: 1px solid #444;
}
.sidebar-btn:hover {
    background-color: #333;
    cursor: pointer;
}
.rename-panel {
    display: flex;
    gap: 5px;
    margin-top: 10px;
}
"""
)

with demo:
    gr.Markdown("<script>document.body.classList.add('dark');</script>")
    gr.Markdown("## 🤖 Makine Titreşim Asistanı\nGPT benzeri sohbet deneyimi yaşayın.")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                yeni_btn = gr.Button("➕ Yeni Sohbet")
                tema_html = gr.HTML("""
                    <button onclick="
                        const b = document.body;
                        b.classList.toggle('dark');
                        b.classList.toggle('light');
                    "
                    style='padding:8px;border-radius:6px;background:#222;color:white;border:1px solid #444;'>
                        🌙 Tema Değiştir
                    </button>
                """)

            with gr.Column(elem_id="sidebar-scroll"):
                sohbet_buttons = [gr.Button(visible=False, elem_classes="sidebar-btn") for _ in range(MAX_BUTTONS)]

        with gr.Column(scale=8):
            sohbet_ekrani = gr.Chatbot(label="Sohbet", show_copy_button=True, type="messages")
            mesaj_kutusu = gr.Textbox(placeholder="Bir mesaj yazın...", show_label=False)
            gonder_btn = gr.Button("Gönder")
            temizle_btn = gr.Button("Temizle")

            gr.Markdown("### Seçilen Sohbet İşlemleri")
            with gr.Row(elem_classes="rename-panel"):
                ad_input = gr.Textbox(label="Yeni Ad", scale=2)
                ad_btn = gr.Button("✏️ Adı Güncelle", scale=1)
                sil_btn = gr.Button("🗑️ Sil", scale=1)

    secili_sohbet = gr.State("")
    sohbet_id_listesi = gr.State(list(chat_histories.keys()))

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

    sil_btn.click(fn=sil_sohbet, inputs=[secili_sohbet],
                  outputs=[sohbet_id_listesi, sohbet_ekrani, secili_sohbet])\
        .then(fn=update_sidebar, inputs=[sohbet_id_listesi], outputs=sohbet_buttons)

    ad_btn.click(fn=yeniden_adlandir,
                 inputs=[secili_sohbet, ad_input],
                 outputs=[sohbet_id_listesi, sohbet_ekrani, secili_sohbet])\
        .then(fn=update_sidebar, inputs=[sohbet_id_listesi], outputs=sohbet_buttons)

    for btn in sohbet_buttons:
        btn.click(fn=load_chat, inputs=[btn],
                  outputs=[sohbet_ekrani, mesaj_kutusu, secili_sohbet, ad_input])  # Add ad_input to outputs

demo.launch()
# sil butonu anlık siliyor, sayfa yenilendiğinde geri geliyor
