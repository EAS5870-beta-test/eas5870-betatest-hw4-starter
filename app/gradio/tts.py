from typing import Tuple
import gradio as gr
from app import service

def build(translated_text_state: gr.State, target_lang_state: gr.State):
    """Build the Text-to-Speech tab UI"""

    with gr.Row():
        # Left column: Inputs
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text or use translation from MT tab...",
                lines=8,
                max_lines=15
            )

            use_translation_btn = gr.Button(
                "← Use Translation from MT Tab",
                variant="secondary"
            )

            with gr.Row():
                lang_dropdown = gr.Dropdown(
                    label="Language",
                    choices=service.get_available_languages(),
                    value="en",
                    scale=1
                )

                voice_dropdown = gr.Dropdown(
                    label="Voice",
                    choices=service.get_available_voices('en'),
                    value="af_heart",
                    scale=1
                )

            speed_slider = gr.Slider(
                label="Speed",
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1
            )

            generate_btn = gr.Button("Generate Audio", variant="primary", size="lg")

        # Right column: Output
        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="filepath",
                interactive=False
            )

            status_text = gr.Textbox(
                label="Status",
                value="Ready to generate audio",
                interactive=False,
                lines=2
            )

    # Load translation and update language to match MT target
    def use_translation(trans, mt_lang):
        lang_mapping = {
            'en': 'en', 'es': 'es', 'fr': 'fr', 'it': 'it',
            'pt': 'pt', 'zh': 'zh', 'ja': 'ja', 'hi': 'hi'
        }
        tts_lang = lang_mapping.get(mt_lang, 'en')
        voices = service.get_available_voices(tts_lang)
        default_voice = voices[0][1] if voices else "af_heart"
        return (
            gr.update(value=trans),
            gr.update(value=tts_lang),
            gr.update(choices=voices, value=default_voice)
        )

    use_translation_btn.click(
        use_translation,
        inputs=[translated_text_state, target_lang_state],
        outputs=[text_input, lang_dropdown, voice_dropdown],
        api_name=False
    )

    # Update voices when language changes manually
    def update_voices(lang):
        voices = service.get_available_voices(lang)
        default_voice = voices[0][1] if voices else "af_heart"
        return gr.update(choices=voices, value=default_voice)

    lang_dropdown.change(
        update_voices,
        inputs=[lang_dropdown],
        outputs=[voice_dropdown],
        api_name=False
    )

    # Generate audio
    generate_btn.click(
        on_generate,
        inputs=[text_input, voice_dropdown, lang_dropdown, speed_slider],
        outputs=[audio_output, status_text],
        api_name=False,
        queue=False
    )

def on_generate(text: str, voice: str, lang: str, speed: float):
    audio_path, status = service.generate_tts(text, voice=voice, lang=lang, speed=speed)
    return audio_path, status
