import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

zip_path = './digits.zip'  
image_folder = './unzipped_digits/digits'  
model_path = './mnist_model.h5'  

def unzip_files():
    if not os.path.exists(image_folder):
        print('Разархивирование файла digits.zip...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('./unzipped_digits')
        print('Файлы успешно разархивированы в папку unzipped_digits/digits.')
    else:
        print('Папка с изображениями уже существует, пропускаем разархивирование.')

def load_or_train_model():
    if os.path.exists(model_path):
        print('Загрузка существующей модели...')
        model = tf.keras.models.load_model(model_path)
        print('Модель загружена!')
    else:
        print('Модель не найдена, начинается обучение...')
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        print('Компиляция и обучение модели...')
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
        model.save(model_path)
        print('Модель обучена и сохранена в формате .h5')
    return model

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  
        return img_array
    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {e}")
        return None

def recognize_digits(model):
    digit_counts = [0] * 10

    if os.path.exists(image_folder):
        files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(files)
        print(f'Обнаружено изображений для обработки: {total_files}')

        for i, file in enumerate(files):
            file_path = os.path.join(image_folder, file)
            print(f'[{i+1}/{total_files}] Обработка файла: {file_path}')

            img_array = preprocess_image(file_path)

            if img_array is not None:
                prediction = model.predict(img_array)
                predicted_digit = np.argmax(prediction)

                print(f'Распознанная цифра: {predicted_digit}')
                digit_counts[predicted_digit] += 1

        print(f'\nМассив для отправки: {digit_counts}')
    else:
        print(f'Папка {image_folder} не найдена. Убедитесь, что путь указан правильно.')

def main():
    unzip_files()                 
    model = load_or_train_model()  
    recognize_digits(model)       

if __name__ == "__main__":
    main()
