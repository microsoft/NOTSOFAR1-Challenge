import whisper
from whisper.normalizers import EnglishTextNormalizer
from jiwer import wer

model = whisper.load_model("large-v2")
result = model.transcribe("example.wav")
normalizer = EnglishTextNormalizer()

ref = open('ref.txt', 'r')
ref_text = ref.readlines()[0]

error = wer(normalizer(result["text"]), normalizer(result["text"]))
print(f'WER = {error}%')
