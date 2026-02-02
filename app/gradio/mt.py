from typing import Tuple
import gradio as gr
from app import service

def build(translated_text_state: gr.State, target_lang_state: gr.State):
    """Build the Machine Translation tab UI"""

    with gr.Row():
        # Left column: Inputs
        with gr.Column(scale=1):
            src_lang = gr.Dropdown(
                choices=service.SUPPORTED_LANGS,
                value="en",
                label="Source Language",
                info="Language of the input text"
            )

            input_text = gr.Textbox(
                label="Text to Translate",
                placeholder="Enter text to translate...",
                lines=10,
                max_lines=20
            )

            tgt_lang = gr.Dropdown(
                choices=service.SUPPORTED_LANGS,
                value="es",
                label="Target Language",
                info="Language to translate into"
            )

            translate_btn = gr.Button("Translate", variant="primary", size="lg")

        # Right column: Output
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Translation Result",
                lines=10,
                max_lines=20,
                interactive=False
            )

            status_text = gr.Textbox(
                label="Status",
                value="Ready to translate",
                interactive=False,
                lines=2
            )

    # Event handlers - updates shared states for cross-tab workflow
    def on_translate(text: str, src: str, tgt: str):
        translation, status = service.translate_text(text, src, tgt)
        return translation, status, translation, tgt

    translate_btn.click(
        on_translate,
        inputs=[input_text, src_lang, tgt_lang],
        outputs=[output_text, status_text, translated_text_state, target_lang_state],
        api_name=False,
        queue=False
    )
