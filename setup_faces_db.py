# -*- coding: utf-8 -*-
"""
نظام بسيط للتعرف على الوجوه
- الموقع: يحفظ الصور في faces_db/
- النظارة: تقرأ الصور من faces_db/
- كل صورة تُحفظ باسم الشخص: name.jpg
- الـ embedding يُحفظ في: name_embedding.pkl
"""

import os
import json

# إعدادات المجلد المشترك
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
FACES_DB_DIR = os.path.join(PROJECT_ROOT, 'faces_db')

# إنشاء المجلد
if not os.path.exists(FACES_DB_DIR):
    os.makedirs(FACES_DB_DIR)
    print(f"[OK] Created: {FACES_DB_DIR}")

# إنشاء ملف تعريفي
config_file = os.path.join(FACES_DB_DIR, 'README.txt')
with open(config_file, 'w', encoding='utf-8') as f:
    f.write("""
==============================================
    مجلد الوجوه المشترك (Shared Faces Database)
==============================================

هذا المجلد يحتوي على جميع الوجوه المحفوظة:
- الموقع (Dashboard) يحفظ الصور هنا
- النظارة الذكية تقرأ من هنا

بنية الملفات:
--------------
محمد.jpg                    # صورة الوجه
محمد_embedding.pkl          # الميزات (Facenet512 embedding)

كل شخص له ملفان فقط!
""")

print("\n[OK] Setup complete!")
print(f"[FOLDER] Shared folder: {FACES_DB_DIR}")
print("\nNow:")
print("1. Website will save images here")
print("2. Smart glasses will read from here")
print("3. No complex database needed!")
