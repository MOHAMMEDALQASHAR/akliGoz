from gtts import gTTS
import os

print("⏳ جاري الاتصال بجوجل وتحميل الصوت...")

try:
    # 1. إنشاء ملف الصوت
    tts = gTTS(text="Merhaba, sistem çalışıyor", lang='tr')
    tts.save("test_sound.mp3")
    print("✅ تم تحميل الملف بنجاح.")

    # 2. محاولة التشغيل باستخدام ffplay (الأكثر ضماناً)
    print("🔊 جاري التشغيل...")
    exit_code = os.system("ffplay -nodisp -autoexit -loglevel quiet test_sound.mp3")
    
    if exit_code == 0:
        print("✅ تم التشغيل بنجاح! هل سمعت الصوت؟")
    else:
        print("❌ فشل التشغيل. تأكد أن سماعات الجهاز تعمل.")

except Exception as e:
    print(f"❌ خطأ: {e}")
    print("تأكد أن الإنترنت متصل!")
