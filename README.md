<div align="center">

# 🩺 Kidney Disease Classification

### نظام تصنيف صور الكلى باستخدام التعلم العميق

تطبيق ويب عربي لتصنيف صور الكلى إلى أربع فئات، مع إعدادات منظمة وتقارير تدريب وتقييم.

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?logo=flask)
![Classes](https://img.shields.io/badge/Classes-4-168aad)

[نظرة عامة](#-نظرة-عامة) ·
[النتائج](#-نتائج-التقييم) ·
[التشغيل](#-التثبيت-والتشغيل) ·
[هيكل المشروع](#-هيكل-المشروع) ·
[التقرير التفصيلي](docs/PROJECT_REPORT.md)

</div>

---

## 📌 نظرة عامة

يهدف المشروع إلى تصنيف صور الكلى إلى إحدى الفئات الأربع التالية:

| الفئة البرمجية | الوصف |
|---|---|
| **Cyst** | كيس |
| **Normal** | طبيعي |
| **Stone** | حصى |
| **Tumor** | ورم |

يتضمن المشروع نموذجًا أساسيًا، وتجربة ضبط دقيق (Fine-tuning)، ووحدة تنبؤ، وواجهة ويب تستخدم Flask لرفع الصور وعرض نتائج النموذج.
