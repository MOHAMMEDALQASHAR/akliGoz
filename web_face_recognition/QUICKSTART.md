# دليل التشغيل السريع - Quick Start Guide

## طريقة 1: استخدام السكريبت التلقائي (موصى به)

```powershell
cd web_face_recognition
.\run.ps1
```

هذا السكريبت سيقوم بـ:
- إنشاء بيئة افتراضية
- تثبيت جميع المكتبات
- إنشاء المجلدات الضرورية
- تشغيل التطبيق

---

## طريقة 2: التشغيل اليدوي

### 1. انتقل لمجلد المشروع
```bash
cd web_face_recognition
```

### 2. ثبت المكتبات
```bash
pip install -r requirements.txt
```

### 3. شغل التطبيق
```bash
python app.py
```

### 4. افتح المتصفح
افتح: `http://localhost:5000`

---

## ملاحظات مهمة

### ✅ بدون Google OAuth
يمكنك استخدام التطبيق **بدون إعداد Google OAuth** عن طريق:
- إنشاء حساب عادي بكلمة مرور
- لن يعمل زر "تسجيل الدخول بـ Google" ولكن الوظائف الأخرى ستعمل بشكل كامل

### ⚙️ مع Google OAuth (اختياري)
لتفعيل تسجيل الدخول بـ Google:

1. اذهب إلى: https://console.cloud.google.com/
2. أنشئ مشروع جديد
3. فعّل Google+ API
4. أنشئ OAuth 2.0 credentials
5. أضف Redirect URI: `http://localhost:5000/login/google/callback`
6. في `app.py` استبدل:
   ```python
   client_id='YOUR_GOOGLE_CLIENT_ID'
   client_secret='YOUR_GOOGLE_CLIENT_SECRET'
   ```

---

## الاستخدام السريع

### 1️⃣ التسجيل
- اضغط "إنشاء حساب جديد"
- أدخل بياناتك
- سجل دخول

### 2️⃣ إضافة وجه
- شغل الكاميرا
- اكتب الاسم
- التقط صورة
- احفظ

### 3️⃣ التعرف
- شغل الكاميرا
- اضغط "التعرف على الوجه"
- سيظهر الاسم تلقائياً!

---

## مشاكل شائعة

### الكاميرا لا تعمل؟
- امنح المتصفح صلاحية الكاميرا
- جرب متصفح آخر (Chrome موصى به)

### خطأ في تثبيت face_recognition؟
**Windows:**
```bash
pip install cmake
pip install dlib
pip install face-recognition
```

**إذا استمر الخطأ:**
قم بتحميل wheel file من:
https://github.com/z-mahmud22/Dlib_Windows_Python3.x

---

## الحل البديل (بدون face_recognition)

إذا واجهت صعوبة في تثبيت `face_recognition`، يمكن استخدام OpenCV فقط للتعرف البسيط.
راجع ملف README.md للتفاصيل.

---

تاريخ: 2025-12-12
