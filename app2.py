#bu ekran app.py dosyası için tasarım dosyasıdır
import gradio as gr

# Gerçek inference fonksiyonuna gerek duymadan sahte cevap ver
def generate_answer(user_question: str) -> str:
    return f"🧠 (Mock Cevap): '{user_question}' sorusu alındı. Bu, test cevaplarıdır."

# Chat ekranı UI
chat_history = []

def chat_interface(user_input, history):
    response = generate_answer(user_input)
    history = history + [[user_input, response]]
    return history, history

with gr.Blocks(title="Makine Titreşim Asistanı (UI Test)") as demo:
    gr.Markdown("""
    # 🤖 Makine Titreşim Asistanı (Mock Modu)
    Gerçek model olmadan sadece arayüzü test ediyorsunuz.
    """)

    chatbot = gr.Chatbot(label="Sohbet", height=400)

    with gr.Row():
        with gr.Column(scale=8):
            msg = gr.Textbox(placeholder="Sorunuzu yazın...", label="Mesajınızı girin", lines=1, autofocus=True)
        with gr.Column(scale=1):
            clear = gr.Button("Temizle")

    state = gr.State([])

    msg.submit(chat_interface, inputs=[msg, state], outputs=[chatbot, state])
    clear.click(lambda: ([], []), outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch()
