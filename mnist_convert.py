import tensorflow as tf

print('Загрузка данных MNIST...')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Данные MNIST загружены.')
print(f'Форма x_train: {x_train.shape}, Форма y_train: {y_train.shape}')

x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

print('Данные нормализованы.')
print(f'Новая форма x_train: {x_train.shape}')

# Создание модели
print('\nСоздание модели...')
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

print('Модель создана:')
model.summary()

print('\nКомпиляция модели...')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print('Модель скомпилирована.')

print('\nНачало обучения модели...')
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
print('\nОбучение модели завершено.')

print('\nСохранение модели в формате .h5...')
model.save('mnist_model.h5')
print('Модель сохранена в формате .h5')
