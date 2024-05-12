import librosa
import numpy as np

def reduzir_ruido(audio, sr):
  """
  Reduz o ruído de fundo do áudio utilizando a biblioteca librosa.
  """
  audio_filtrado = librosa.decompose.nn_filter(audio, sr=sr)
  return audio_filtrado

def normalizar_audio(audio):
  """
  Normaliza o volume do áudio para um intervalo padrão.
  """
  audio_normalizado = librosa.util.normalize(audio)
  return audio_normalizado

# Outras funções de pré-processamento podem ser adicionadas aqui.
