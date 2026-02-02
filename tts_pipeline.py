#!/usr/bin/env python3
import argparse
import uuid
import os
import json
import hashlib
import shutil
from gradio_client import Client

TTS_API_URL = "http://localhost:7863/"

# Supported languages for translation
SUPPORTED_LANGS = ["en", "es", "fr", "hi", "it", "ja", "pt", "zh"]

# Initialize Gradio client
tts_client = Client(TTS_API_URL)

def call_tts(text, output_file_path, voice="af_heart", lang="en", speed=1.0):
    """Call TTS API and save audio to file"""
    try:
        # Call Gradio API - returns (filepath, status)
        audio_path, status = tts_client.predict(
            text=text,
            voice=voice,
            lang=lang,
            speed=speed,
            api_name="/generate_tts"
        )

        if not audio_path:
            print(f"TTS failed: {status}")
            return False

        # Copy generated file to output location
        shutil.copy(audio_path, output_file_path)
        return True
    except Exception as e:
        print(f"TTS error: {e}")
        return False

def call_mt(text, src_lang, tgt_lang):
    """Call Machine Translation API and return translated text"""
    try:
        # Call Gradio API for translation
        translation, status = tts_client.predict(
            text=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            api_name="/translate_text"
        )

        if not translation:
            print(f"MT failed: {status}")
            return None

        return translation
    except Exception as e:
        print(f"MT error: {e}")
        return None

def build_audio_json(basepath, english_to_speech_map, language="en"):
    """Update the audio data JSON file.
    
    Args:
        basepath: Base path for the project
        english_to_speech_map: Dict mapping (english_text, speech_text) -> asset_name
        language: Target language code
    """
    json_path = os.path.join(basepath, "Audio", language, "audio_data_table.json")

    gamepath = basepath.split("Content")[-1].replace("\\", "/")

    data = []

    for (english_text, speech_text), asset in english_to_speech_map.items():
        row = dict()
        row['EnglishText'] = english_text
        row['SpeechText'] = speech_text
        row['SoundWavePath'] = f"/Script/Engine.SoundWave'/Game{gamepath}Audio/{language}/{asset}.{asset}'"
        row['Name'] = uuid.uuid4().hex
        data.append(row)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def get_ink_dialogue_lines(ink_file_path):
    return tts_client.predict(ink_file_path, api_name="/parse_ink")

def get_asset_and_filename_for_text(audio_directory, text):
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    asset_name = f"shopkeeper_{text_hash}"
    wav_file = os.path.join(audio_directory, f"{asset_name}.wav")
    return {
        "asset_name": asset_name,
        "wav_file": wav_file
    }

def run_pipeline(basepath, target_lang="en", voice="af_heart"):
    print(f"Running pipeline with base path: {basepath}")
    print(f"Target language: {target_lang}")
    print(f"Voice: {voice}")
    
    audio_directory = os.path.join(basepath, "Audio", target_lang)
    print(f"Temporary WAV directory: {audio_directory}")
    os.makedirs(audio_directory, exist_ok=True)

    # Write a pipeline to process ink files and generate TTS audio using the functions above to fill this data structure
    text_to_asset_map: dict[tuple[str, str], str] = {}
    
    # When calling call_tts, use the function get_asset_and_filename_for_text to get the output filename and asset name.
    # call_tss will take the wav file path as an argument to save the audio, but you will need to build the mapping of 
    # (english_text, speech_text) -> asset_name using the asset_name instead of the wav file path for the Unreal import to work.

    build_audio_json(basepath, text_to_asset_map, target_lang)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TTS Pipeline - Generate speech audio from ink dialogue files"
    )
    parser.add_argument(
        "base_path",
        help="Base path containing your Level and Ink directory (e.g., /path/to/YourProject/Content/Maps/PalmerStation/)"
    )
    parser.add_argument(
        "-l", "--language",
        choices=SUPPORTED_LANGS,
        default="en",
        help=f"Target language for translation (default: en). Supported: {', '.join(SUPPORTED_LANGS)}"
    )
    parser.add_argument(
        "-v", "--voice",
        default="af_heart",
        help="Voice name for TTS (default: af_heart). See service.py VOICES dict for available voices per language."
    )
    
    args = parser.parse_args()
    
    print(f"Base path: {args.base_path}")
    print(f"Target language: {args.language}")
    print(f"Voice: {args.voice}")
    run_pipeline(args.base_path, args.language, args.voice)
