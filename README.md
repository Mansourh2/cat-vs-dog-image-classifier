في هذا المشروع، قمت باستخدام Teachable Machine من Google لتدريب نموذج تصنيف صور بسيط باستخدام فئتين: قطة و كلب. بعد تدريب النموذج، استخدمت Google Colab لتشغيل الكود الذي يقوم بتحميل النموذج المدرب، معالجة الصور المدخلة، ثم إجراء التنبؤ بالفئة المتوقعة.

الخطوات التي اتبعتها في المشروع:

1. تدريب النموذج:
	•	قمت باستخدام Teachable Machine من Google
قمت بتدريب النموذج على فئتين فقط: قطة و كلب، باستخدام صور تم جمعها لهذه الغاية.
	•	بعد الانتهاء من تدريب النموذج، قمت بتصدير النموذج بتنسيق Keras (في الملف keras_model.h5).

2. رفع الملفات إلى Google Colab:
	•	في البداية، رفعت الملفات الأربعة (النموذج المدرب، ملف التسميات، الصورة المدخلة، ولقطة الشاشة) عبر Google Colab باستخدام الكود التالي
from google.colab import files
uploaded = files.upload()
عند تنفيذ هذا الكود، يتم فتح نافذة لتحديد الملفات من جهازك المحلي ورفعها إلى Google Colab
TensorFlow:
	•	بعد رفع الملفات، قمت بتثبيت TensorFlow باستخدام الإصدار المطلوب الذي كان 2.12.1، باستخدام الأمر التالي
!pip install tensorflow==2.12.1
4. كتابة كود Python لتحميل النموذج وتنفيذ التنبؤ:
	•	بعد التأكد من رفع الملفات وتثبيت TensorFlow، كتبت الكود في Google Colab لتحميل النموذج المدرب، معالجة الصورة المدخلة، ثم إجراء التنبؤ.
	•	تحميل النموذج المدرب
model = tf.keras.models.load_model('/content/keras_model.h5')
تحميل التسميات من ملف labels.txt
with open('/content/labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
معالجة الصورة المدخلة باستخدام دالة preprocess_image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')  # تأكد من أن الصورة بنظام RGB
    img = img.resize((224, 224))  # تغيير الحجم إلى 224x224
    img_array = np.array(img, dtype=np.float32)  # تحويل الصورة إلى مصفوفة
    img_array = img_array / 255.0  # تطبيع القيم
    return np.expand_dims(img_array, axis=0)  # إضافة بعد إضافي لجعلها [1, 224, 224, 3]

	•	إجراء التنبؤ بالفئة المتوقعة للصورة
image_path = '/content/test_image.jpg'
input_image = preprocess_image(image_path)
predictions = model.predict(input_image)
predicted_class = class_names[np.argmax(predictions)]
print(f"Predicted class: {predicted_class}")
5. تشغيل الكود في Google Colab:
	•	بعد رفع الملفات وتثبيت TensorFlow، قمت بتشغيل الكود في Google Colab باستخدام Shift + Enter أو عبر الضغط على زر “Run” في الخلية.
	•	نتيجة التنبؤ ظهرت كما يلي
Predicted class: cat
