const tf = require('@tensorflow/tfjs-node');
const mnist = require('mnist');

console.log('Версия TensorFlow.js:', tf.version.tfjs);

function loadMnistData() {
  const { training } = mnist.set(60000, 0);

  const xTrain = [];
  const yTrain = [];

  for (let i = 0; i < training.length; i++) {
    const image = training[i].input;
    const label = training[i].output.indexOf(1);

    const reshapedImage = tf.tensor3d(image, [28, 28, 1]);
    xTrain.push(reshapedImage);
    yTrain.push(label);
  }

  const xTrainTensor = tf.stack(xTrain);
  const yTrainTensor = tf.tensor1d(yTrain, 'float32');

  console.log('Данные MNIST загружены.');
  console.log(
    'Форма x_train:',
    xTrainTensor.shape,
    ', Форма y_train:',
    yTrainTensor.shape
  );

  return { xTrainTensor, yTrainTensor };
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy'],
  });

  console.log('\nСтруктура модели:');
  model.summary();

  return model;
}

async function trainModel(model, xTrain, yTrain) {
  console.log('\nНачало обучения модели...');

  await model.fit(xTrain, yTrain, {
    epochs: 5,
    batchSize: 128,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Эпоха ${epoch + 1}: точность = ${(logs.acc * 100).toFixed(
            2
          )}%, потеря = ${logs.loss.toFixed(4)}`
        );
      },
    },
  });

  console.log('\nОбучение модели завершено.');
}

async function saveModel(model) {
  const savePath = 'file://./mnist_model';
  await model.save(savePath);
  console.log('Модель сохранена в формате TensorFlow.js');
}

async function runModelTraining() {
  const { xTrainTensor, yTrainTensor } = loadMnistData();
  const model = createModel();

  await trainModel(model, xTrainTensor, yTrainTensor);
  await saveModel(model);

  xTrainTensor.dispose();
  yTrainTensor.dispose();
}

if (require.main === module) {
  (async () => {
    await runModelTraining();
  })();
}

module.exports = { runModelTraining };
