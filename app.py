import gradio as gr
from inference import generate_answer

def run_interface(user_question: str) -> str:
    """
    Gradio’dan gelen metni alır, generate_answer() ile işleyip döner.
    """
    return generate_answer(user_question)

demo = gr.Interface(
    fn=run_interface,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Makineyle ilgili sorunuzu yazın…",
        label="user_question"
    ),
    outputs=gr.Textbox(
        label="Asistan Cevabı"
    ),
    title="Makine Titreşim Asistanı",
    description="Meta LLaMA + LoRA + FAISS ile eğitilmiş Türkçe endüstriyel asistan",
)

if __name__ == "__main__":
    # paylaşmak isterseniz share=True ekleyebilirsiniz: demo.launch(share=True)
    demo.launch()
