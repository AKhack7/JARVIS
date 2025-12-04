import asyncio
import threading
import os
import edge_tts
import pygame

VOICE = "en-AU-WilliamNeural"
BUFFER_SIZE = 1024

def remove_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error removing file: {e}")

def play_audio(file_path):
    try:
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)

        pygame.mixer.quit()
    except Exception as e:
        print(f"Audio error : {e}")

async def amain(TEXT, output_file) -> None:
    try:
        tts = edge_tts.Communicate(TEXT, VOICE)
        await tts.save(output_file)

        thread = threading.Thread(target=play_audio, args=(output_file,))
        thread.start()
        thread.join()
    except Exception as e:
        print(f"TTS error : {e}")
    finally:
        remove_file(output_file)

def speak(TEXT, output_file=None):
    if output_file is None:
        output_file = os.path.join(os.getcwd(), "speech.mp3")
    asyncio.run(amain(TEXT, output_file))

# Test
if __name__ == "__main__":
    speak("Welcome to the world of Jarvis")
