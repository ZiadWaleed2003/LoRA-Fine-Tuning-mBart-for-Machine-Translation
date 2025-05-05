import gradio as gr
from translator import Translator

# a single instance of the Translator
translator = Translator()

def translate_text(arabic_text):
    if not arabic_text.strip():
        return "Please enter some Arabic text to translate."
    
    try:
        translated_text = translator.translate(arabic_text)
        return translated_text
    except Exception as e:
        return f"Translation error: {str(e)}"

# Create a Gradio interface
with gr.Blocks(title="Arabic to English Translator") as demo:
    gr.Markdown("# Arabic to English Translator")
    gr.Markdown("This app translates Arabic text to English using the mBart-LoRA model.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Arabic Text",
                placeholder="أدخل النص العربي هنا...",
                lines=5
            )
        
        with gr.Column():
            output_text = gr.Textbox(
                label="English Translation",
                lines=5
            )
    
    translate_btn = gr.Button("Translate")
    translate_btn.click(
        fn=translate_text,
        inputs=input_text,
        outputs=output_text
    )
    
    gr.Markdown("### Examples")
    examples = gr.Examples(
        examples=[
            ["أهلاً بك"],
            ["كيف حالك؟"],
            ["أنا سعيد بلقائك"],
            ["مرحباً بكم في تطبيق الترجمة من العربية إلى الإنجليزية"]
        ],
        inputs=input_text
    )


if __name__ == "__main__":
    demo.launch()

