const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const Jimp = require('jimp');
const AdmZip = require('adm-zip');
const { exec } = require('child_process');

console.log('Версия TensorFlow.js:', tf.version.tfjs);

const modelPath = 'file://./mnist_model/model.json';
const imageFolder = './unzipped_digits/digits';
const zipPath = './digits.zip';
let digitCounts = Array(10).fill(0);

function unzipFiles() {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(imageFolder)) {
      console.log('Разархивирование файла digits.zip...');
      const zip = new AdmZip(zipPath);
      zip.extractAllTo('./unzipped_digits', true);
      console.log(
        'Файлы успешно разархивированы в папку unzipped_digits/digits.'
      );
    } else {
      console.log(
        'Папка с изображениями уже существует, пропускаем разархивирование.'
      );
    }
    resolve();
  });
}

function trainModel() {
  return new Promise((resolve, reject) => {
    console.log('Запуск обучения модели...');
    exec('node create_mnist_model.js', (error, stdout, stderr) => {
      if (error) {
        console.error(`Ошибка при обучении модели: ${error.message}`);
        return reject(error);
      }
      if (stderr) {
        console.warn(`Предупреждение во время обучения: ${stderr}`);
      }
      console.log(`Обучение завершено:\n${stdout}`);
      resolve();
    });
  });
}

async function preprocessImage(imagePath) {
  try {
    const image = await Jimp.read(imagePath);
    if (!image) return null;

    image.resize(28, 28).grayscale();
    const imgArray = new Float32Array(28 * 28);
    image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
      imgArray[y * 28 + x] = image.bitmap.data[idx] / 255.0;
    });

    return tf.tensor4d(imgArray, [1, 28, 28, 1]);
  } catch (err) {
    console.log(`Ошибка при обработке ${imagePath}: ${err.message}`);
    return null;
  }
}

async function recognizeDigits() {
  try {
    if (!fs.existsSync('./mnist_model/model.json')) {
      console.log('Модель не найдена. Проверьте, было ли завершено обучение.');
      return;
    }

    const model = await tf.loadLayersModel(modelPath);
    console.log('Модель успешно загружена.');

    const files = fs
      .readdirSync(imageFolder)
      .filter((file) => file.match(/\.(png|jpg|jpeg)$/i));
    const totalFiles = files.length;
    console.log(`Обнаружено изображений в папке: ${totalFiles}`);

    for (let i = 0; i < totalFiles; i++) {
      const file = files[i];
      const filePath = path.join(imageFolder, file);
      console.log(`[${i + 1}/${totalFiles}] Обработка файла: ${filePath}`);

      const imgTensor = await preprocessImage(filePath);
      if (imgTensor) {
        const prediction = model.predict(imgTensor);
        const predictedDigit = prediction.argMax(1).dataSync()[0];

        console.log(`Предсказание для изображения ${i + 1}: ${predictedDigit}`);
        digitCounts[predictedDigit]++;
      } else {
        console.log(`Не удалось загрузить изображение: ${filePath}`);
      }
    }

    console.log(
      `\nИтоговый массив для отправки: ${JSON.stringify(digitCounts)}`
    );
  } catch (err) {
    console.error('Ошибка в recognizeDigits():', err);
  }
}

(async () => {
  try {
    await unzipFiles();
    await trainModel();
    await recognizeDigits();
  } catch (err) {
    console.error('Ошибка в основном процессе:', err);
  }
})();
