import os

from pyannote.audio import Pipeline

hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("PYANNOTE_TOKEN")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token,
)

print("Loaded pyannote pipeline:", pipeline is not None)
print("Token provided:", bool(hf_token))
