import gradio as gr

from app.gradio.mt import build as build_mt
from app.gradio.tts import build as build_tts
from app.service import (
    warmup,
    translate_text,
    generate_tts,
    parse_ink,
    get_available_voices,
    get_available_languages,
)

with gr.Blocks(title="TTS and MT") as demo:
    gr.Markdown("""### Text To Speech and Machine Translation""")

    # Shared states for cross-tab workflow
    translated_text = gr.State("")
    target_lang = gr.State("en")

    with gr.Tabs():
        with gr.Tab("Machine Translation"):
            build_mt(translated_text, target_lang)
        with gr.Tab("Text-to-Speech"):
            build_tts(translated_text, target_lang)

    # Expose service functions via API
    gr.api(translate_text, api_name="translate_text")
    gr.api(generate_tts, api_name="generate_tts")
    gr.api(parse_ink, api_name="parse_ink")
    gr.api(get_available_voices, api_name="get_available_voices")
    gr.api(get_available_languages, api_name="get_available_languages")

if __name__ == "__main__":
    warmup()
    print("\n" + "="*50)
    print("Launching Gradio on http://0.0.0.0:7863")
    print("="*50 + "\n")
    demo.launch(server_name="0.0.0.0", server_port=7863)

