import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("--- الأصوات المتاحة ---")
for voice in voices:
    # نبحث عن أي صوت فيه كلمة Turkish أو tr
    if 'tr' in voice.id or 'turkish' in voice.name.lower():
        print(f"✅ وجدنا صوتاً تركياً: {voice.id}")
    else:
        print(f"Sound ID: {voice.id}")
