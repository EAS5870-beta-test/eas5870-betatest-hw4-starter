"""
TTS and MT Service Module
Business logic for text-to-speech (Kokoro) and machine translation (NLLB-200)
"""

# === Machine Translation Section ===
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple, List, Optional, Union
import sys
import os

# Force CUDA to use default stream for all operations
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    # Disable CUDA graph capture which can cause issues with threading
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

# MT Global state
mt_model = None
mt_tokenizer = None
mt_device = None

LANG_CODE_MAP = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "pt": "por_Latn",
    "zh": "zho_Hans",
}

SUPPORTED_LANGS = list(LANG_CODE_MAP.keys())


def initialize_mt_model() -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, torch.device]:
    """Load NLLB-200 model for machine translation"""
    global mt_model, mt_tokenizer, mt_device

    if mt_model is not None:
        return mt_model, mt_tokenizer, mt_device

    print("Loading NLLB-200 model...")

    mt_model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M",
        torch_dtype=torch.float16
    )
    mt_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M",
    )

    mt_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mt_model.to(mt_device)
    mt_model.eval()

    print(f"Model loaded on {mt_device}")

    return mt_model, mt_tokenizer, mt_device


def translate_text(
    text: Union[str, List[str]],
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 16
) -> Tuple[Union[str, List[str]], str]:
    """
    Translate text from source language to target language.

    Args:
        text: Single text string or list of texts to translate
        src_lang: Source language code (e.g., 'en', 'es')
        tgt_lang: Target language code (e.g., 'en', 'es')
        batch_size: Number of texts to process at once (default 16)

    Returns:
        (translation(s), status): Single string or list of strings, plus status message.
        For batch input, failed items return empty strings.
    """
    global mt_model, mt_tokenizer, mt_device

    # Detect if input is a list
    is_batch = isinstance(text, list)

    if is_batch:
        return _translate_text_batch(text, src_lang, tgt_lang, batch_size)

    if not text or not text.strip():
        return "", "Error: Please enter some text to translate"

    if src_lang not in LANG_CODE_MAP:
        return "", f"Error: Unsupported source language: {src_lang}"

    if tgt_lang not in LANG_CODE_MAP:
        return "", f"Error: Unsupported target language: {tgt_lang}"

    if src_lang == tgt_lang:
        return "", "Error: Source and target languages must be different"

    if mt_model is None:
        try:
            initialize_mt_model()
        except Exception as e:
            return "", f"Error initializing model: {str(e)}"

    try:
        src_lang_code = LANG_CODE_MAP[src_lang]
        tgt_lang_code = LANG_CODE_MAP[tgt_lang]
        mt_tokenizer.src_lang = src_lang_code

        with torch.no_grad():
            inputs = mt_tokenizer(
                text,
                return_tensors='pt',
                padding=True,
            ).to(mt_device)

            target_lang_id = mt_tokenizer.convert_tokens_to_ids(tgt_lang_code)
            generated_tokens = mt_model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_new_tokens=100,
                num_beams=1,
                do_sample=False
            )
        output = mt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return output, "Translation successful!"

    except Exception as e:
        error_msg = f"Error during translation: {str(e)}"
        print(error_msg, file=sys.stderr)
        return "", error_msg


def _translate_text_batch(
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 16
) -> Tuple[List[str], str]:
    """Internal batch translation helper."""
    global mt_model, mt_tokenizer, mt_device

    if not texts:
        return [], "Error: Empty text list provided"

    if src_lang not in LANG_CODE_MAP:
        return [""] * len(texts), f"Error: Unsupported source language: {src_lang}"

    if tgt_lang not in LANG_CODE_MAP:
        return [""] * len(texts), f"Error: Unsupported target language: {tgt_lang}"

    if src_lang == tgt_lang:
        return [""] * len(texts), "Error: Source and target languages must be different"

    if mt_model is None:
        try:
            initialize_mt_model()
        except Exception as e:
            return [""] * len(texts), f"Error initializing model: {str(e)}"

    src_lang_code = LANG_CODE_MAP[src_lang]
    tgt_lang_code = LANG_CODE_MAP[tgt_lang]
    target_lang_id = mt_tokenizer.convert_tokens_to_ids(tgt_lang_code)

    results = [""] * len(texts)
    errors = []
    successful = 0

    # Process in batches
    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]

        # Track valid texts and their indices within this batch
        valid_indices = []
        valid_texts = []
        for i, t in enumerate(batch_texts):
            if t and t.strip():
                valid_indices.append(i)
                valid_texts.append(t)

        if not valid_texts:
            continue

        try:
            mt_tokenizer.src_lang = src_lang_code

            with torch.no_grad():
                inputs = mt_tokenizer(
                    valid_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(mt_device) for k, v in inputs.items()}

                generated_tokens = mt_model.generate(
                    **inputs,
                    forced_bos_token_id=target_lang_id,
                    max_new_tokens=100,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True,
                    do_sample=False
                )

            outputs = mt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # Map outputs back to original indices
            for idx, output in zip(valid_indices, outputs):
                results[batch_start + idx] = output
                successful += 1

        except Exception as e:
            error_msg = f"Error in batch {batch_start}-{batch_end}: {str(e)}"
            print(error_msg, file=sys.stderr)
            errors.append(error_msg)

    # Build status message
    if successful == len(texts):
        status = f"Translation successful! ({successful} texts)"
    elif successful > 0:
        status = f"Partial success: {successful}/{len(texts)} texts translated"
        if errors:
            status += f". Errors: {'; '.join(errors)}"
    else:
        status = f"Translation failed. Errors: {'; '.join(errors)}" if errors else "No valid texts to translate"

    return results, status


# === Text-to-Speech Section ===
import numpy as np
import soundfile as sf
import tempfile
from kokoro import KPipeline

tts_pipelines = {}

LANG_MAP = {
    'en': 'a',
    'en-us': 'a',
    'en-gb': 'b',
    'es': 'e',
    'fr': 'f',
    'hi': 'h',
    'it': 'i',
    'ja': 'j',
    'pt': 'p',
    'zh': 'z',
}

VOICES = {
    'en': [
        ('Heart (Female, Grade A)', 'af_heart'),
        ('Bella (Female, Grade A-)', 'af_bella'),
        ('Nicole (Female, Grade B-)', 'af_nicole'),
        ('Aoede (Female, Grade C+)', 'af_aoede'),
        ('Kore (Female, Grade C+)', 'af_kore'),
        ('Sarah (Female, Grade C+)', 'af_sarah'),
        ('Alloy (Female, Grade C)', 'af_alloy'),
        ('Nova (Female, Grade C)', 'af_nova'),
        ('Sky (Female, Grade C-)', 'af_sky'),
        ('Fenrir (Male, Grade C+)', 'am_fenrir'),
        ('Michael (Male, Grade C+)', 'am_michael'),
        ('Puck (Male, Grade C+)', 'am_puck'),
    ],
    'en-gb': [
        ('Emma (Female, Grade B-)', 'bf_emma'),
        ('Isabella (Female, Grade C)', 'bf_isabella'),
        ('Alice (Female, Grade D)', 'bf_alice'),
        ('Lily (Female, Grade D)', 'bf_lily'),
        ('George (Male, Grade C)', 'bm_george'),
        ('Fable (Male, Grade C)', 'bm_fable'),
        ('Lewis (Male, Grade D+)', 'bm_lewis'),
        ('Daniel (Male, Grade D)', 'bm_daniel'),
    ],
    'es': [
        ('Dora (Female)', 'ef_dora'),
        ('Alex (Male)', 'em_alex'),
        ('Santa (Male)', 'em_santa'),
    ],
    'fr': [
        ('Siwis (Female, Grade B-)', 'ff_siwis'),
    ],
    'hi': [
        ('Alpha (Female, Grade C)', 'hf_alpha'),
        ('Beta (Female, Grade C)', 'hf_beta'),
        ('Omega (Male, Grade C)', 'hm_omega'),
        ('Psi (Male, Grade C)', 'hm_psi'),
    ],
    'it': [
        ('Sara (Female, Grade C)', 'if_sara'),
        ('Nicola (Male, Grade C)', 'im_nicola'),
    ],
    'ja': [
        ('Alpha (Female, Grade C+)', 'jf_alpha'),
        ('Gongitsune (Female, Grade C)', 'jf_gongitsune'),
        ('Nezumi (Female, Grade C-)', 'jf_nezumi'),
        ('Tebukuro (Female, Grade C)', 'jf_tebukuro'),
        ('Kumo (Male, Grade C-)', 'jm_kumo'),
    ],
    'pt': [
        ('Dora (Female)', 'pf_dora'),
        ('Alex (Male)', 'pm_alex'),
        ('Santa (Male)', 'pm_santa'),
    ],
    'zh': [
        ('Xiaobei (Female, Grade D)', 'zf_xiaobei'),
        ('Xiaoni (Female, Grade D)', 'zf_xiaoni'),
        ('Xiaoxiao (Female, Grade D)', 'zf_xiaoxiao'),
        ('Xiaoyi (Female, Grade D)', 'zf_xiaoyi'),
        ('Yunjian (Male, Grade D)', 'zm_yunjian'),
        ('Yunxi (Male, Grade D)', 'zm_yunxi'),
        ('Yunxia (Male, Grade D)', 'zm_yunxia'),
        ('Yunyang (Male, Grade D)', 'zm_yunyang'),
    ],
}

SAMPLE_RATE = 24000


def get_tts_pipeline(lang: str = 'en') -> KPipeline:
    """Get or create a KPipeline for the specified language"""
    global tts_pipelines

    if lang not in tts_pipelines:
        kokoro_lang = LANG_MAP.get(lang, 'a')
        tts_pipelines[lang] = KPipeline(lang_code=kokoro_lang)
        print(f"Pipeline for {lang} is running on: {tts_pipelines[lang].model.device}")
        
    return tts_pipelines[lang]


def get_available_voices(lang: str = 'en') -> List[Tuple[str, str]]:
    """Get available voices for a language"""
    return VOICES.get(lang, VOICES['en'])


def get_available_languages() -> List[Tuple[str, str]]:
    """Get list of available languages for TTS"""
    return [
        ('English (US)', 'en'),
        ('English (UK)', 'en-gb'),
        ('Spanish', 'es'),
        ('French', 'fr'),
        ('Hindi', 'hi'),
        ('Italian', 'it'),
        ('Japanese', 'ja'),
        ('Portuguese (BR)', 'pt'),
        ('Chinese (Mandarin)', 'zh'),
    ]


def generate_tts(
    text: Union[str, List[str]],
    voice: str = 'af_heart',
    lang: str = 'en',
    speed: float = 1.0,
    batch_size: int = 16
) -> Tuple[Union[Optional[str], List[Optional[str]]], str]:
    """
    Generate TTS audio from text using Kokoro.

    Args:
        text: Single text string or list of texts to synthesize
        voice: Voice ID to use (e.g., 'af_heart')
        lang: Language code (e.g., 'en', 'es')
        speed: Speech speed multiplier (0.5-2.0)
        batch_size: Number of texts to process at once (default 16, advisory for progress)

    Returns:
        (audio_path(s), status): Single path or list of paths, plus status message.
        For batch input, failed items return None.
    """
    # Detect if input is a list
    is_batch = isinstance(text, list)

    if is_batch:
        return _generate_tts_batch(text, voice, lang, speed, batch_size)

    if not text or not text.strip():
        return None, "Error: Please enter some text to synthesize"

    try:
        pipeline = get_tts_pipeline(lang)

        audio_chunks = []
        for i, (graphemes, phonemes, audio) in enumerate(pipeline(text, voice=voice, speed=speed)):
            audio_chunks.append(audio)

        if not audio_chunks:
            return None, "Error: No audio generated"

        full_audio = np.concatenate(audio_chunks)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, full_audio, SAMPLE_RATE)
            output_path = temp_file.name

        return output_path, "Audio generated successfully!"

    except Exception as e:
        error_msg = f"Error generating audio: {str(e)}"
        print(error_msg, file=sys.stderr)
        return None, error_msg


def _generate_tts_batch(
    texts: List[str],
    voice: str = 'af_heart',
    lang: str = 'en',
    speed: float = 1.0,
    batch_size: int = 16
) -> Tuple[List[Optional[str]], str]:
    """Internal batch TTS helper."""
    if not texts:
        return [], "Error: Empty text list provided"

    try:
        pipeline = get_tts_pipeline(lang)
    except Exception as e:
        return [None] * len(texts), f"Error initializing TTS pipeline: {str(e)}"

    results: List[Optional[str]] = [None] * len(texts)
    errors = []
    successful = 0

    for i, t in enumerate(texts):
        if not t or not t.strip():
            continue

        try:
            audio_chunks = []
            for _, (graphemes, phonemes, audio) in enumerate(pipeline(t, voice=voice, speed=speed)):
                audio_chunks.append(audio)

            if not audio_chunks:
                errors.append(f"Text {i}: No audio generated")
                continue

            full_audio = np.concatenate(audio_chunks)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, full_audio, SAMPLE_RATE)
                results[i] = temp_file.name

            successful += 1

        except Exception as e:
            error_msg = f"Text {i}: {str(e)}"
            print(f"Error generating audio for text {i}: {str(e)}", file=sys.stderr)
            errors.append(error_msg)

    # Build status message
    if successful == len(texts):
        status = f"Audio generated successfully! ({successful} files)"
    elif successful > 0:
        status = f"Partial success: {successful}/{len(texts)} audio files generated"
        if errors:
            status += f". Errors: {'; '.join(errors[:3])}"
            if len(errors) > 3:
                status += f" (and {len(errors) - 3} more)"
    else:
        status = f"Audio generation failed. Errors: {'; '.join(errors[:3])}" if errors else "No valid texts to synthesize"
        if len(errors) > 3:
            status += f" (and {len(errors) - 3} more)"

    return results, status


def warmup():
    """Warmup both MT and TTS services"""
    print("Warming up services...")

    print("Loading NLLB-200 translation model...")
    try:
        initialize_mt_model()
        print("Translation model ready!")
    except Exception as e:
        print(f"Warning: Translation model failed to load: {e}", file=sys.stderr)

    print("Loading Kokoro TTS pipeline...")
    try:
        get_tts_pipeline('en')
        print("TTS pipeline ready!")
    except Exception as e:
        print(f"Warning: TTS pipeline failed to load: {e}", file=sys.stderr)

    print("All services ready!")


def parse_ink(ink_file_path: str) -> List[str]:
    """Parse ink file for dialogue lines from shuffle blocks.
    
    Extracts all lines from shuffle blocks that look like:
    {shuffle:
        - "A translucent horror that steals sound, leaving only terror."
        - "It glides unseen, a gelatinous shroud of quiet despair."
        ...
    }
    """
    import re
    
    print(f"Parsing ink file: {ink_file_path}")
    if not os.path.exists(ink_file_path):
        print(f"Ink file not found: {ink_file_path}")
        return []

    with open(ink_file_path, 'r') as f:
        content = f.read()

    # Find all shuffle, cycle, and stopping blocks and extract quoted strings from lines starting with -
    shuffle_pattern = r'\{\s*shuffle:\s*(.*?)\}'
    cycle_pattern = r'\{\s*cycle:\s*(.*?)\}'
    stopping_pattern = r'\{\s*stopping:\s*(.*?)\}'
    shuffle_blocks = re.findall(shuffle_pattern, content, re.DOTALL)
    cycle_blocks = re.findall(cycle_pattern, content, re.DOTALL)
    stopping_blocks = re.findall(stopping_pattern, content, re.DOTALL)

    dialogue_lines = []
    line_pattern = r'-\s*(.+)'
    all_blocks = [*shuffle_blocks, *cycle_blocks, *stopping_blocks]
    for block in all_blocks:
        # Extract all quoted strings from lines starting with -
        quotes = re.findall(line_pattern, block)
        for quote in quotes:
            if quote not in dialogue_lines:
                dialogue_lines.append(quote)

    return dialogue_lines
