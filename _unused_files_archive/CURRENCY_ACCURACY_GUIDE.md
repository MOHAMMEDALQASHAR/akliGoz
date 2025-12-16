# دليل تحسين دقة التعرف على النقود التركية
# Turkish Currency Recognition Accuracy Improvement Guide

## الخطوة 1: إعادة تدريب النموذج بإعدادات محسّنة
## Step 1: Retrain Model with Improved Settings

### أ) تحضير البيانات (إذا لم يتم من قبل)
### A) Prepare Data (if not done already)

```bash
python prepare_classification_data.py
```

### ب) تدريب النموذج المحسّن (50 حقبة بدلاً من 10)
### B) Train Improved Model (50 epochs instead of 10)

```bash
python train_classifier_advanced.py
```

**ملاحظة:** هذا سيستغرق وقتاً أطول (حوالي 30-60 دقيقة) لكن النتائج ستكون أدق بكثير!
**Note:** This will take longer (about 30-60 minutes) but results will be much more accurate!

---

## الخطوة 2: تحديث البرنامج الرئيسي لاستخدام النموذج الجديد
## Step 2: Update Main Program to Use New Model

بعد اكتمال التدريب، افتح `main_glasses.py` وابحث عن السطر:
After training completes, open `main_glasses.py` and find the line:

```python
custom_model_path = "runs/classify/currency_cls/weights/best.pt"
```

واستبدله بـ / Replace it with:

```python
custom_model_path = "runs/classify/currency_cls_advanced/weights/best.pt"
```

---

## الخطوة 3: استخدام الكود المحسّن (اختياري)
## Step 3: Use Enhanced Code (Optional)

### لتطبيق التحسينات على identify_currency:
### To apply improvements to identify_currency:

1. افتح `main_glasses.py`
2. أضف import في البداية:

```python
from currency_recognition_enhanced import preprocess_currency_image, identify_currency_enhanced
```

3. في class SmartGlassesAssistant، استبدل دالة identify_currency بالاستدعاء المحسّن:

```python
def identify_currency(self, frame):
    identify_currency_enhanced(frame, self.currency_model, self.voice, self.currency_matcher)
```

---

## التحسينات المُطبقة / Improvements Applied

### 1. **تدريب أفضل (Better Training)**
   - ✅ 50 حقبة بدلاً من 10 (50 epochs instead of 10)
   - ✅ حجم صورة أكبر: 320px (Larger image size: 320px)
   - ✅ Data Augmentation متقدم (Advanced data augmentation)
   - ✅ AdamW optimizer (أفضل من SGD)
   - ✅ Label smoothing لمنع الثقة المفرطة

### 2. **معالجة صور محسّنة (Better Image Processing)**
   - ✅ Denoising - إزالة التشويش
   - ✅ CLAHE - تحسين التباين
   - ✅ Resize محسّن (Optimized resize)

### 3. **تقييم متعدد (Multi-Angle Evaluation)**
   - ✅ الصورة الأصلية (Original image)
   - ✅  الصورة المقلوبة (Flipped image)
   - ✅ Crop مركزي (Center crop)
   - ✅ اختيار أفضل نتيجة (Best result selection)

### 4. **عتبة ثقة أعلى (Higher Confidence Threshold)**
   - ✅ 0.60 للتأكيد الكامل (0.60 for full confidence)
   - ✅ 0.45-0.60 للاحتمالية (0.45-0.60 for probability)
   - ✅ أقل من 0.45: SIFT fallback

---

## النتائج المتوقعة / Expected Results

### قبل التحسين (Before):
- الدقة: ~75-85%
- Confidence threshold: 0.50
- معالجة بسيطة (Simple processing)

### بعد التحسين (After):
- الدقة المتوقعة: ~90-95%+
- Confidence threshold: 0.60
- معالجة متقدمة + تقييم متعدد

---

## نصائح إضافية / Additional Tips

1. **الإضاءة (Lighting):** تأكد من وجود إضاءة جيدة عند استخدام الكاميرا

2. **المسافة (Distance):** احتفظ بالنقود على بُعد 20-40 سم من الكاميرا

3. **الاستقرار (Stability):** حاول إبقاء الكاميرا ثابتة أثناء المسح

4. **جودة النقود (Bill Quality):** النقود النظيفة والمسطحة تُعطي نتائج أفضل

5. **زاوية الرؤية (Viewing Angle):** حاول أن يكون المنظور مباشراً (ليس من زاوية جانبية حادة)

---

## استكشاف الأخطاء / Troubleshooting

### المشكلة: النموذج يعطي نتائج خاطئة لبعض الفئات
**الحل:** أعد التدريب مع المزيد من epochs أو أضف المزيد من صور التدريب لتلك الفئات

### المشكلة: التدريب بطيء جداً
**الحل:** قلل عدد الـ epochs إلى 30، أو قلل batch size إلى 8

### المشكلة: خطأ "Out of Memory" أثناء التدريب
**الحل:** في `train_classifier_advanced.py`، قلل batch size من 16 إلى 8 أو 4

---

## معلومات الأداء / Performance Information

**التدريب على CPU:**
- الزمن المتوقع: 45-90 دقيقة
- استهلاك الذاكرة: ~2-4 GB RAM

**التدريب على GPU:**
- الزمن المتوقع: 10-20 دقيقة
- أسرع بـ 3-5 مرات

**الاستخدام (Inference):**
- زمنكل تعرف: ~0.5-1.5 ثانية (مع 3 محاولات)
- سريع بما يكفي للاستخدام الفعلي

---

## الخلاصة / Summary

لتحسين الدقة بشكل ملحوظ:

1. ✅ شغّل `python train_classifier_advanced.py`
2. ✅ انتظر حتى ينتهي التدريب
3. ✅ حدّث المسار في `main_glasses.py`
4. ✅ شغّل البرنامج واستمتع بالدقة المحسّنة!

---

## للتواصل والدعم / Support

إذا واجهت أي مشاكل، راجع:
- ملفات السجل في `runs/classify/currency_cls_advanced/`
- رسائل الخطأ في Console
- تأكد من أن جميع المكتبات مثبتة

---

**تاريخ الإنشاء:** 2025-12-12
**الإصدار:** 2.0 (Enhanced)
