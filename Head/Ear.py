import speech_recognition as sr
import os
import threading
from mtranslate import translate
from colorama import Fore, Style, init

init(autoreset=True)  # automatically reset style after each print

def print_loop():
    while True:
        print(Fore.LIGHTGREEN_EX + "I am Listening...", end="\r", flush=True)

def Trans_hindi_to_english(txt):
    english_txt = translate(txt, to_language="en-in")
    return english_txt

def listen():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000
    recognizer.dynamic_energy_threshold = False
    recognizer.pause_threshold = 0.3

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)

        while True:
            print(Fore.LIGHTGREEN_EX + "I am Listening...", end="\r", flush=True)
            try:
                audio = recognizer.listen(source)
                print(Fore.YELLOW + "Got it, recognizing...", end="\r", flush=True)
                recognized_txt = recognizer.recognize_google(audio, language="hi-IN").lower()

                if recognized_txt:
                    translated_txt = Trans_hindi_to_english(recognized_txt)
                    print(Fore.BLUE + "Mr Zeno : " + translated_txt)
                    return translated_txt
                else:
                    return ""

            except sr.UnknownValueError:
                print(Fore.RED + "Didn't catch that...", end="\r", flush=True)
            except Exception as e:
                print(Fore.RED + f"Error : {e}", flush=True)

def main():
    while True:
        listen()

if __name__ == "__main__":
    main()
