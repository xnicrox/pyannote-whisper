import whisper
from pyannote.audio import Pipeline
from pyannote.audio import Audio
from pyannote_whisper.utils import diarize_text

# Is necesary a token of https://huggingface.co/settings/tokens
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")

model = whisper.load_model("tiny")
diarization_result = pipeline("data/2023-07-10_22-01-04_mono.wav")

from pyannote.audio import Audio
audio = Audio(sample_rate=16000, mono=True)
audio_file = "data/2023-07-10_22-01-04_mono.wav"

# Create text file
#fileDocument=open("datos.txt","w",encoding="utf-8") 
 
# Iteration
for segment, _, speaker in diarization_result.itertracks(yield_label=True):
    waveform, sample_rate = audio.crop(audio_file, segment)
    text = model.transcribe(waveform.squeeze().numpy())["text"]
    print(f"{segment.start:.2f}s {segment.end:.2f}s {speaker}: {text}")
    #fileDocument.write(f"{segment.start:.2f}s {segment.end:.2f}s {speaker}: {text}")
    
#fileDocument.close() 