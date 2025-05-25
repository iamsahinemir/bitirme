import gradio as gr

# Sahte cevap üretici
def generate_answer(message):
    return f"I received: '{message}' — but this is a mock response."

def chat_interface(message, history):
    response = generate_answer(message)
    history.append((message, response))
    return history, ""

with gr.Blocks(theme=gr.themes.Soft(), title="My Chatbot") as demo:
    gr.Markdown("## 🧠 My Chatbot\nEnter text to start chatting.")

    chatbot = gr.Chatbot(label="Chatbot", bubble_full_width=False, show_copy_button=True)
    msg_box = gr.Textbox(placeholder="Type a message...", show_label=False, scale=9)
    state = gr.State([])

    with gr.Row():
        send_btn = gr.Button("Submit", scale=1)
    
    with gr.Row():
        retry = gr.Button("🔁 Retry")
        undo = gr.Button("↩️ Undo")
        clear = gr.Button("🗑️ Clear")

    # Mesaj gönderildiğinde sohbet güncellenir, input temizlenir
    msg_box.submit(chat_interface, inputs=[msg_box, state], outputs=[chatbot, msg_box])
    send_btn.click(chat_interface, inputs=[msg_box, state], outputs=[chatbot, msg_box])

    # Buton fonksiyonları
    clear.click(fn=lambda: ([], ""), outputs=[chatbot, msg_box])
    undo.click(fn=lambda h: (h[:-1], ""), inputs=[state], outputs=[chatbot, msg_box])
    retry.click(fn=lambda h: h, inputs=[state], outputs=chatbot)  # sadece örnek

if __name__ == "__main__":
    demo.launch()
