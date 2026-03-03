from transformers import pipeline
import scipy

pipe = pipeline("text-to-audio", model="facebook/musicgen-small")
musica = pipe("classic soul track with deep bass and saxophone", forward_params={"max_new_tokens": 256})
audio_limpio = musica["audio"].squeeze()
scipy.io.wavfile.write("prueba_soul.wav", rate=musica["sampling_rate"], data=audio_limpio)