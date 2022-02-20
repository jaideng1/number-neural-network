const {ungzip} = require('node-gzip');
const fs = require('fs');
const path = require('path');

const Matrix = require('./matrix.js');
const { NeuralNetwork, DeeperNeuralNetwork } = require('./nn.js');

const trainingLabelsPath = './training-data/mnist/train-labels-idx1-ubyte.gz';
const trainingImagesPath = './training-data/mnist/train-images-idx3-ubyte.gz';

async function loadTrainingImages() {
    let trainingLabelData = fs.readFileSync(trainingLabelsPath);

    let decompressed = await ungzip(trainingLabelData);

    let trainingLabels = decompressed.toString('binary');

    let label_count = Buffer.from(trainingLabels.slice(4, 8), 'binary').readUInt32BE(0);
    let label_data = trainingLabels.slice(8);

    let labels = [];

    for (let i = 0; i < label_count; i++) {
        labels.push(Buffer.from(label_data[i]).readUInt8(0));
    }

    let trainingImageData = fs.readFileSync(trainingImagesPath);

    decompressed = await ungzip(trainingImageData);

    let trainingImages = decompressed.toString('binary');

    let image_count = Buffer.from(trainingImages.slice(4, 8), 'binary').readUInt32BE(0);
    let rows = Buffer.from(trainingImages.slice(8, 12), 'binary').readUInt32BE(0);
    let cols = Buffer.from(trainingImages.slice(12, 16), 'binary').readUInt32BE(0);

    let image_data = trainingImages.slice(16);
    
    let images = [];

    for (let i = 0; i < image_count; i++) {
        let image = [];

        for (let j = 0; j < rows * cols; j++) {
            image.push(Buffer.from(image_data[i * rows * cols + j]).readUInt8(0) / 255);
        }

        images.push(image);
    }

    return {
        labels: labels,
        images: images
    };
}

function trainAI() {
    console.log("Loading training data...");
    const beforeTime = new Date().getTime();

    loadTrainingImages().then(async (data) => {
        const nowTime = new Date().getTime();

        let nn = new DeeperNeuralNetwork(784, 100, 10);
        /*nn.loadFile("/training_directory/folder/your_training_data_here.json")*/

        console.log("Finished loading data, created empty neural network. Took " + Math.floor((nowTime - beforeTime) / 1000) + " seconds to load the data.");

        let inputs = [];
        let expectedOutputs = [];

        for (let i = 0; i < data.labels.length; i++) {
            inputs.push(data.images[i]);
            expectedOutputs.push([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

            expectedOutputs[expectedOutputs.length - 1][data.labels[i]] = 1;
        }
        
        const epochs = 1000000; //Change the number of epochs/how many times it runs the data and improves it.

        console.log("Prepared inputs and outputs, running for " + epochs + " epochs.");
        console.log("Using " + data.labels.length + " training samples.");

        //console.log(typeof inputs, inputs.length, inputs[0].length, typeof expectedOutputs, expectedOutputs.length, expectedOutputs[0].length);

        const pathFolder = path.join(__dirname, `/training-data/${new Date().getTime()}_data/`);

        if (!fs.existsSync(pathFolder)) {
            fs.mkdirSync(pathFolder);
            console.log("Created new folder for training data.");
        }

        await nn.fit(inputs, expectedOutputs, {
            epochs,
            saving: {
                pathFolder,
                saveEvery: 100000
            }
        });

        console.log("Finished training AI.");

        nn.save(pathFolder + "finished.json");

        console.log("Saved data! Finished.");
    }).catch((e) => {
        console.error("Error while running training.")
        console.error(e);
    });
}

async function testAI(amount) {
    let nn = new DeeperNeuralNetwork(784, 100, 10);

    await nn.loadFile(path.join(__dirname, "/training-data/1645320483198_data/finished.json"));

    loadTrainingImages().then(async (data) => {
        console.log("Loaded training data. Testing a random sample size of " + amount);

        let inputs = [];
        let expectedOutputs = [];

        for (let i = 0; i < amount; i++) {
            let randomNumber = Math.floor(Math.random() * data.labels.length);

            inputs.push(data.images[randomNumber]);
            expectedOutputs.push([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

            expectedOutputs[expectedOutputs.length - 1][data.labels[randomNumber]] = 1;
        }

        let amountCorrect = 0;

        for (let i = 0; i < inputs.length; i++) {
            let outputs = await nn.predict(inputs[i]);

            let max = 0;
            let maxIndex = 0;

            for (let j = 0; j < outputs.length; j++) {
                if (outputs[j] > max) {
                    max = outputs[j];
                    maxIndex = j;
                }
            }

            if (maxIndex == expectedOutputs[i].indexOf(1)) {
                amountCorrect++;
            }

            console.log("Expected: " + expectedOutputs[i].indexOf(1) + " Got: " + maxIndex);
        }

        console.log("Finished testing AI. " + amountCorrect + " out of " + amount + " were correct.");
    });
}

testAI(100); //or trainAI()
