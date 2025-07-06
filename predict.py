import tensorflow as tf
import numpy as np
from PIL import Image

# تحميل النموذج
model = tf.keras.models.load_model('/content/keras_model.h5')

# تحميل التسميات
with open('/content/labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# معالجة الصورة
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')  # تأكد من أن الصورة بنظام RGB
    img = img.resize((224, 224))  # تغيير الحجم إلى 224x224
    img_array = np.array(img, dtype=np.float32)  # تحويل الصورة إلى مصفوفة
    img_array = img_array / 255.0  # تطبيع القيم
    return np.expand_dims(img_array, axis=0)  # إضافة بعد إضافي لجعلها [1, 224, 224, 3]

# تحديد مسار الصورة
image_path = '/content/test_image.jpg'

# التنبؤ بالصورة
input_image = preprocess_image(image_path)

# التنبؤ بالفئة
predictions = model.predict(input_image)
predicted_class = class_names[np.argmax(predictions)]

# طباعة النتيجة
print(f"Predicted class: {predicted_class}")
