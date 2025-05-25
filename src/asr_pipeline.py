"""
Example 
CUDA_VISIBLE_DEVICES=1 python src/asr_pipeline.py --input_dir /data/user/standup/audio/pl/ --output_dir /data/user/standup/transcript/pl/distil-whisper-large-v3-pl/ --language pl --model_id Aspik101/distil-whisper-large-v3-pl
or 
python src/asr_pipeline.py --input_dir /data/user/standup/audio/cs/ --output_dir /data/user/standup/transcript/cs/whisper-large-v3-czech/ --language cs --model_id Cem13/whisper-large-v3-czech
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set to -1 for CPU

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import glob
import os
import time
import argparse
import json
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
import pandas as pd

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch audio transcription using Whisper')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for transcription JSON files')
    parser.add_argument('--language', type=str, required=True,
                        help='Language of the audio files (e.g., "fr" for French)')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Batch size for processing (default: 12)')
    parser.add_argument('--model_id', type=str,
                        default="openai/whisper-large-v3-turbo",
                        help='Whisper model ID to use')
    parser.add_argument('--return_timestamps', type=str, default='word', choices=['word', 'chunk'],
                        help='Language of the audio files (e.g., "fr" for French)') 
    return parser.parse_args()

def setup_model(model_id: str) -> tuple:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logging.info(f"Using device: {device}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=1,  # TODO: fix this one
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def read_json_asr(path_asr):
    df = pd.read_json(path_asr)['chunks'].apply(pd.Series)
    df[['t0', 't1']] = pd.DataFrame(df['timestamp'].to_list(), index=df.index)
    # Drop the original timestamp column
    df = df.drop(columns='timestamp')
    df['t'] = df['t1'] - df['t0']
    return df

def segment_audio_files(audio_files: List[str], 
                        thresh: float = 1.2, 
                        latence: float = 0.5) -> List[str]:
    """
    Use the ASR output, in particular the long words, to segment the audio file, in order to get more accurate timestamps.
    In the end, it was a test and did not work better. 
    """

    from pydub import AudioSegment

    audio_files_seg = []

    for fn in audio_files:
        lang, fname = fn.split('/')[-2:]
        fname = fname.split('.')[0]
        fn_asr = glob.glob(f"/home/user/data/standup/transcript/{lang}/*/{fname}.json")[0]
        df = read_json_asr(fn_asr)
        df = df[df.t>thresh]

        os.makedirs(os.path.split(fn)[0] + "/seg/", exist_ok=True)

        # Load the audio file
        audio = AudioSegment.from_file(fn)
        start_ms = 0
        # Loop through timestamps and cut segments
        for i, row in df.iterrows():
            # pydub works in milliseconds
            end_ms = int((row['t1']-latence) * 1000)
            if start_ms != end_ms: 
                segment = audio[start_ms:end_ms]
                path_seg = os.path.split(fn)[0] + f"/seg/{i:04d}_{fname}.wav"
                segment.export(path_seg, format="wav")
                audio_files_seg.append(path_seg)
                start_ms = end_ms
                
    return audio_files_seg


def get_audio_files(input_dir: str) -> List[str]:
    files = glob.glob(os.path.join(input_dir, "*"))
    # Filter for common audio extensions
    # audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    audio_files = [f for f in files if os.path.isfile(f) and
                   os.path.splitext(f)[1].lower() in audio_extensions]
    return audio_files

def process_batch(pipe, audio_files: List[str], output_dir: str,
                  language: str, batch_size: int, model_id: str, return_timestamps: str):
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(0, len(audio_files), batch_size), desc="Processing batches"):
        batch = audio_files[i:i + batch_size]
        start_time = time.time()
        
        # Filter out already processed files
        batch_to_process = []
        for audio_file in batch:
            output_file = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(audio_file))[0]}.json"
            )
            if not os.path.exists(output_file):
                batch_to_process.append(audio_file)
            else:
                logging.info(f"Skipping already processed file: {audio_file}")
        
        if not batch_to_process:
            continue
        
        try:
            results = pipe(batch_to_process,
                            return_timestamps=return_timestamps,
                            # return_timestamps="chunk", # This is very problematic if I used this...
                            generate_kwargs={"language": language})
            elapsed_time = time.time() - start_time
            logging.info(f"Batch processed in {elapsed_time:.2f} seconds")
            #from utils_crisper import adjust_pauses_for_hf_pipeline_output
            #print(results)
            #results = adjust_pauses_for_hf_pipeline_output(results)
            #print(results)
            
            # Save results for each file in the batch
            for audio_file, result in zip(batch_to_process, results):
                output_data = {
                    "audio_file": os.path.basename(audio_file),
                    "full_text": result["text"],
                    "chunks": result["chunks"],
                    "processing_time": elapsed_time,
                    "language": language,
                    "model_id": model_id,
                }
                output_file = os.path.join(
                    output_dir,
                    f"{os.path.splitext(os.path.basename(audio_file))[0]}.json"
                )
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                logging.info(f"Saved transcription to {output_file}")
        
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            for audio_file in batch_to_process:
                logging.error(f"Failed to process file: {audio_file}")

def main():
    setup_logging()
    args = parse_arguments()
    logging.info("Initializing model...")
    pipe = setup_model(args.model_id)
    logging.info(f"Scanning directory: {args.input_dir}")
    audio_files = get_audio_files(args.input_dir)
    logging.info(f"Found {len(audio_files)} audio files")
    if not audio_files:
        logging.error("No audio files found in the input directory")
        return
    
    if False: # to change by arg
        audio_files = segment_audio_files(audio_files)
        logging.info(f"Segmented in {len(audio_files)} audio files")

    logging.info("Starting batch processing...")
    process_batch(pipe,
                  audio_files,
                  args.output_dir,
                  args.language,
                  args.batch_size,
                  args.model_id, 
                  args.return_timestamps)
    logging.info("Processing completed successfully")

if __name__ == "__main__":
    main()