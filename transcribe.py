import os
import json
import whisper
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

RAW_AUDIO_DIR = "raw_audio"
OUTPUT_DIR = "clips"
MIN_DURATION = 3000
MAX_DURATION = 10000
SILENCE_THRESH = -40
MIN_SILENCE_LEN = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
model = whisper.load_model("large-v3")

audio_files = [f for f in os.listdir(RAW_AUDIO_DIR) if f.endswith('.wav')]
print(f"{len(audio_files)} files found")

clip_counter = 1
metadata = []

for audio_file in audio_files:
    audio_path = os.path.join(RAW_AUDIO_DIR, audio_file)
    print(f"\n{audio_file}")
    
    audio = AudioSegment.from_wav(audio_path)
    segments = detect_nonsilent(audio, min_silence_len=MIN_SILENCE_LEN, 
                               silence_thresh=SILENCE_THRESH)
    
    print(f"{len(segments)} segments")
    
    for start, end in segments:
        duration = end - start
        
        if MIN_DURATION <= duration <= MAX_DURATION:
            clip = audio[start:end]
            clip = clip.set_frame_rate(16000).set_channels(1)
            
            filename = f"clip_{clip_counter:03d}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)
            clip.export(filepath, format="wav")
            
            result = model.transcribe(filepath)
            text = result["text"].strip()
            
            print(f"{filename}: {text[:40]}...")
            
            metadata.append({
                'file_name': filename,
                'text': text,
                'duration_seconds': round(duration / 1000, 2),
                'source_file': audio_file
            })
            
            clip_counter += 1

with open('transcriptions.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\n{len(metadata)} clips done")