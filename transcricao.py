import speech_recognition as sr
import pyaudio
from processamento_audio import reduzir_ruido, normalizar_audio
from modelo_transcricao import Transcritor

def capturar_audio():
  """
  Captura áudio em tempo real utilizando a biblioteca SpeechRecognition.
  """
  reconhecedor = sr.Recognizer()
  with sr.Microphone() as source:
    print("Ouvindo...")
    audio = reconhecedor.listen(source)
    return audio

def main():
  """
  Lógica principal do programa.
  """
  # Inicializa o modelo de transcrição
  transcritor = Transcritor()
  while True:
    # Captura áudio
    audio = capturar_audio()
    try:
      # Converte o áudio para formato utilizável pelo modelo
      audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
      sr = audio.sample_rate

      # Pré-processa o áudio
      audio_filtrado = reduzir_ruido(audio_data, sr)
      audio_normalizado = normalizar_audio(audio_filtrado)

      # Transcreve o áudio
      transcricao = transcritor.transcrever(audio_normalizado, sr)

      # Exibe a transcrição
      print("Transcrição:", transcricao)

    except sr.UnknownValueError:
      print("Não foi possível entender o áudio.")
    except sr.RequestError as e:
      print(f"Erro na solicitação: {e}")

if __name__ == "__main__":
  main()
