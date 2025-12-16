import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
print("Testing sound...")
engine.say("Hello Amr, system is ready.")
engine.runAndWait()
print("Done.")
