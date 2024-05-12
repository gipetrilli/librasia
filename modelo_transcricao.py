from transformers import pipeline

class Transcritor:
  def __init__(self):
    """
    Inicializa o modelo de transcrição BERT.
    """
    self.modelo = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53")

  def transcrever(self, audio, sr):
    """
    Transcreve o áudio utilizando o modelo BERT.
    """
    transcricao = self.modelo({"array": audio, "sampling_rate": sr})
    return transcricao["text"]
