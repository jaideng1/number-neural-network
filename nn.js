let usingNodeJS = false;
let __Matrix;
let fs = "NOT_FOUND";

try {
    __Matrix = require('./matrix.js');
    fs = require('fs');
    usingNodeJS = true;
    console.log("Detected NodeJS, using ./Matrix.js");
} catch (e) {
    __Matrix = Matrix;
    console.log("NodeJS not found, using browser Matrix");
}

function NewMatrix(a, b) {
    return new __Matrix(a, b);
}

/**
 * Thanks to: https://towardsdatascience.com/understanding-and-implementing-neural-networks-in-java-from-scratch-61421bb6352c
 * for the help!
 * 
 * (also thanks to copilot for a bit of help lol)
 */

class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
        this.learning_rate = 0.01;

        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        this.weights_input_to_hidden = NewMatrix(hiddenNodes, inputNodes);
        this.weights_hidden_to_output = NewMatrix(outputNodes, hiddenNodes);
        this.bias_hidden = NewMatrix(hiddenNodes, 1);
        this.bias_output = NewMatrix(outputNodes, 1);

        this.weights_input_to_hidden.initialize();
        this.weights_hidden_to_output.initialize();
        this.bias_hidden.initialize();
        this.bias_output.initialize();
    }

    predict(x) {
        if (typeof x != 'object') throw "not array";

        let input = __Matrix.fromArray(x);
        let hidden = __Matrix.multiply(this.weights_input_to_hidden, input);
        hidden.add(this.bias_hidden);
        hidden.sigmoid();

        let output = __Matrix.multiply(this.weights_hidden_to_output,hidden);
        output.add(this.bias_output);
        output.sigmoid();

        return output.toArray();
    }

    async train(x, y) {
        //Calculate the predicted output of the neural network
        let input = __Matrix.fromArray(x);
        let hidden = __Matrix.multiply(this.weights_input_to_hidden, input);
        hidden.add(this.bias_hidden);
        hidden.sigmoid();

        let output = __Matrix.multiply(this.weights_hidden_to_output, hidden);
        output.add(this.bias_output);
        output.sigmoid();

        //Get the correct output
        let target = __Matrix.fromArray(y);

        //Calculate the error
        let error = __Matrix.subtract(target, output);
        //Get the gradient of the output
        let gradient = output.dsigmoid();
        //Multiply the error by the gradient
        gradient.multiply(error);
        //Multiply the gradient by the learning rate
        gradient.multiply(this.learning_rate);

        //Calculate how much the hidden layer weights need to change
        let hidden_transposed = __Matrix.transpose(hidden);
        let weights_hidden_output_delta = __Matrix.multiply(gradient, hidden_transposed);

        //Calculate how much the input layer weights need to change
        let weights_hidden_output_transposed = __Matrix.transpose(this.weights_hidden_to_output);
        let hidden_errors = __Matrix.multiply(weights_hidden_output_transposed, error);

        let hidden_gradient = hidden.dsigmoid();
        hidden_gradient.multiply(hidden_errors);
        hidden_gradient.multiply(this.learning_rate);

        let input_transposed = __Matrix.transpose(input);
        let weights_input_hidden_delta = __Matrix.multiply(hidden_gradient, input_transposed);

        //Add the hidden gradient to the input to hidden weights
        this.weights_hidden_to_output.add(weights_hidden_output_delta);
        this.bias_output.add(gradient);

        this.weights_input_to_hidden.add(weights_input_hidden_delta);
        this.bias_hidden.add(hidden_gradient);
    }

    async fit(x, y, epochs) {
        let currentTime = new Date().getTime();
        let delta1000EpochTime = 0;
        let totalTime = 0;

        for (let i = 0; i < epochs; i++)
        {
            let sampleN = Math.floor(Math.random() * x.length);
            await this.train(x[sampleN], y[sampleN]);

            if (i % 1000 == 0) {
                console.log("Current Epoch: " + i);
                if (i == 1000) {
                    console.log("Reached first 1000 epochs! Some data for the rest of the epochs:");
                    const deltaEpochTime = new Date().getTime() - currentTime;
                    delta1000EpochTime = deltaEpochTime;
                    console.log("Time for 1000 epochs: " + Math.floor(deltaEpochTime / 1000) + "s");
                    const timeTotal = deltaEpochTime * (epochs / 1000); //In ms
                    totalTime = timeTotal;
                    console.log("Expected time in total " + Math.floor(timeTotal / 1000) + "s");
                    console.log("Time left: " + Math.floor((timeTotal - deltaEpochTime) / 1000) + "s");
                }

                if (i % 10000 == 0 && i > 1000) {
                    console.log("% through epochs: " + Math.floor((i / epochs) * 100) + "%");
                    console.log("Time taken so far: " + Math.floor((delta1000EpochTime * (i / 1000)) / 1000) + "s");
                    console.log("Expected time in total " + Math.floor(totalTime/1000) + "s");
                    //time in ms - (timePer1000Epochs * time for 1000 epochs)
                    console.log("Time left: " + Math.floor((totalTime - (delta1000EpochTime * (i / 1000))) / 1000) + "s");
                }
            }
        }

        let timeTaken = new Date().getTime() - currentTime;

        //Convert timeTaken to minutes
        let minutes = timeTaken / 60000;

        console.log("Time taken: " + minutes + " minutes.");
    }

    save(fileLocation) {
        if (fs == "NOT_FOUND") throw "Can't save, NodeJS not found";

        let data = {
            "weights_input_to_hidden": {
                array: this.weights_input_to_hidden.toArray(),
                rows: this.weights_input_to_hidden.rows,
                cols: this.weights_input_to_hidden.cols
            },
            "weights_hidden_to_output": {
                array: this.weights_hidden_to_output.toArray(),
                rows: this.weights_hidden_to_output.rows,
                cols: this.weights_hidden_to_output.cols
            },
            "bias_hidden": {
                array: this.bias_hidden.toArray(),
                rows: this.bias_hidden.rows,
                cols: this.bias_hidden.cols
            },
            "bias_output": {
                array: this.bias_output.toArray(),
                rows: this.bias_output.rows,
                cols: this.bias_output.cols
            },
            "ai_info": {
                "learning_rate": this.learning_rate,
                "inputs": this.inputNodes,
                "hidden": this.hiddenNodes,
                "outputs": this.outputNodes,
                "activation": "sigmoid"
            }
        };

        fs.writeFileSync(fileLocation, JSON.stringify(data));
    }

    loadFile(fileLocation) {
        if (fs == "NOT_FOUND") throw "Can't load by file, NodeJS not found";
        
        this.load(fs.readFileSync(fileLocation));
    }

    loadURL(path) {
        if (!path.startsWith("/")) path = "/" + path;
        if (path.replaceAll("\\", "").includes("node_modules")) throw "ayo? o_O node_modules? nahh";
        if (usingNodeJS) throw "Can't use NodeJS with this method since fetch is not supported";

        fetch(path).then(response => response.json()).then(data => this.load(data, true));
    }

    load(dataString, parsed=false) {
        let data = parsed ? dataString : JSON.parse(dataString);

        console.log("Loading AI...")

        this.weights_input_to_hidden = __Matrix.fromArray(data.weights_input_to_hidden.array, data.weights_input_to_hidden.rows, data.weights_input_to_hidden.cols);
        this.weights_hidden_to_output = __Matrix.fromArray(data.weights_hidden_to_output.array, data.weights_hidden_to_output.rows, data.weights_hidden_to_output.cols);
        this.bias_hidden = __Matrix.fromArray(data.bias_hidden.array, data.bias_hidden.rows, data.bias_hidden.cols);
        this.bias_output = __Matrix.fromArray(data.bias_output.array, data.bias_output.rows, data.bias_output.cols);

        this.learning_rate = data.ai_info.learning_rate;
        this.inputNodes = data.ai_info.inputs;
        this.hiddenNodes = data.ai_info.hidden;
        this.outputNodes = data.ai_info.outputs;
        
        console.log("Loaded AI from JSON!");
        console.log(`AI info:`);
        console.log(`Learning rate: ${this.learning_rate}`);
        console.log(`Input nodes: ${this.inputNodes}`);
        console.log(`Hidden nodes: ${this.hiddenNodes}`);
        console.log(`Output nodes: ${this.outputNodes}`);
    }
}

class DeeperNeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
        this.learning_rate = 0.01;

        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        this.weights_input_to_hidden = NewMatrix(this.hiddenNodes, this.inputNodes);
        this.weights_hidden_to_hidden = NewMatrix(hiddenNodes, hiddenNodes);
        this.weights_hidden_to_output = NewMatrix(outputNodes, hiddenNodes);
        
        this.bias_hidden1 = NewMatrix(hiddenNodes, 1);
        this.bias_hidden2 = NewMatrix(hiddenNodes, 1);
        this.bias_output = NewMatrix(outputNodes, 1);

        this.weights_input_to_hidden.initialize();
        this.weights_hidden_to_hidden.initialize();
        this.weights_hidden_to_output.initialize();
        this.bias_hidden1.initialize();
        this.bias_hidden2.initialize();
        this.bias_output.initialize();
    }

    predict(x) {
        if (typeof x != "object") throw "x must be an array";

        let input = __Matrix.fromArray(x);

        let hidden1 = __Matrix.multiply(this.weights_input_to_hidden, input);
        hidden1.add(this.bias_hidden1);
        hidden1.sigmoid();

        let hidden2 = __Matrix.multiply(this.weights_hidden_to_hidden, hidden1);
        hidden2.add(this.bias_hidden2);
        hidden2.sigmoid();

        let output = __Matrix.multiply(this.weights_hidden_to_output, hidden2);
        output.add(this.bias_output);
        output.sigmoid();

        return output.toArray();
    }

    save(fileLocation) {
        if (fs == "NOT_FOUND") throw "Can't save, NodeJS not found";

        let data = {
            "weights_input_to_hidden": {
                array: this.weights_input_to_hidden.toArray(),
                rows: this.weights_input_to_hidden.rows,
                cols: this.weights_input_to_hidden.cols
            },
            "weights_hidden_to_hidden": {
                array: this.weights_hidden_to_hidden.toArray(),
                rows: this.weights_hidden_to_hidden.rows,
                cols: this.weights_hidden_to_hidden.cols
            },
            "weights_hidden_to_output": {
                array: this.weights_hidden_to_output.toArray(),
                rows: this.weights_hidden_to_output.rows,
                cols: this.weights_hidden_to_output.cols
            },
            "bias_hidden1": {
                array: this.bias_hidden1.toArray(),
                rows: this.bias_hidden1.rows,
                cols: this.bias_hidden1.cols
            },
            "bias_hidden2": {
                array: this.bias_hidden2.toArray(),
                rows: this.bias_hidden2.rows,
                cols: this.bias_hidden2.cols
            },
            "bias_output": {
                array: this.bias_output.toArray(),
                rows: this.bias_output.rows,
                cols: this.bias_output.cols
            },
            "ai_info": {
                "learning_rate": this.learning_rate,
                "inputs": this.inputNodes,
                "hidden": this.hiddenNodes,
                "outputs": this.outputNodes,
                "activation": "sigmoid"
            }
        };

        fs.writeFileSync(fileLocation, JSON.stringify(data));
    }

    loadURL(path) {
        if (!path.startsWith("/")) path = "/" + path;
        if (path.replaceAll("\\", "").includes("node_modules")) throw "ayo? o_O node_modules? nahh";
        if (usingNodeJS) throw "Can't use NodeJS with this method since fetch is not supported";

        fetch(path).then(response => response.json()).then(data => this.load(data, true));
    }

    loadFile(fileLocation) {
        if (fs == "NOT_FOUND") throw "Can't load by file, NodeJS not found";
        
        this.load(fs.readFileSync(fileLocation));
    }

    load(dataString, parsed=false) {
        let data = parsed ? dataString : JSON.parse(dataString);

        this.weights_input_to_hidden = __Matrix.fromArray(data.weights_input_to_hidden.array, data.weights_input_to_hidden.rows, data.weights_input_to_hidden.cols);
        this.weights_hidden_to_hidden = __Matrix.fromArray(data.weights_hidden_to_hidden.array, data.weights_hidden_to_hidden.rows, data.weights_hidden_to_hidden.cols);
        this.weights_hidden_to_output = __Matrix.fromArray(data.weights_hidden_to_output.array, data.weights_hidden_to_output.rows, data.weights_hidden_to_output.cols);

        this.bias_hidden1 = __Matrix.fromArray(data.bias_hidden1.array, data.bias_hidden1.rows, data.bias_hidden1.cols);
        this.bias_hidden2 = __Matrix.fromArray(data.bias_hidden2.array, data.bias_hidden2.rows, data.bias_hidden2.cols);
        this.bias_output = __Matrix.fromArray(data.bias_output.array, data.bias_output.rows, data.bias_output.cols);

        this.learning_rate = data.ai_info.learning_rate;
        this.inputNodes = data.ai_info.inputs;
        this.hiddenNodes = data.ai_info.hidden;
        this.outputNodes = data.ai_info.outputs;

        console.log("Loaded AI from JSON!");
        console.log(`AI info:`);
        console.log(`Learning rate: ${this.learning_rate}`);
        console.log(`Input nodes: ${this.inputNodes}`);
        console.log(`Hidden nodes: ${this.hiddenNodes} (Hidden layers: 2)`);
        console.log(`Output nodes: ${this.outputNodes}`);
    }

    async train(x, y) {
        if (typeof x != "object") throw "x must be an array";
        if (typeof y != "object") throw "y must be an array";

        //Calculate the output
        let input = __Matrix.fromArray(x);

        let hidden1 = __Matrix.multiply(this.weights_input_to_hidden, input);
        hidden1.add(this.bias_hidden1);
        hidden1.sigmoid();

        let hidden2 = __Matrix.multiply(this.weights_hidden_to_hidden, hidden1);
        hidden2.add(this.bias_hidden2);
        hidden2.sigmoid();

        let output = __Matrix.multiply(this.weights_hidden_to_output, hidden2);
        output.add(this.bias_output);
        output.sigmoid();

        //Calculate the error
        let target = __Matrix.fromArray(y);
        let output_errors = __Matrix.subtract(target, output);
        
        //Calculate the gradient of the error
        let gradients = __Matrix.map(output, (x) => {
            return x * (1 - x);
        });
        gradients.multiply(output_errors);
        
        gradients.multiply(this.learning_rate);

        //Calculate deltas
        let hidden_T = __Matrix.transpose(hidden2);
        let weight_ho_deltas = __Matrix.multiply(gradients, hidden_T);

        //Adjust the weights by deltas
        this.weights_hidden_to_output.add(weight_ho_deltas);
        this.bias_output.add(gradients);

        //Calculate the hidden layer errors
        let who_t = __Matrix.transpose(this.weights_hidden_to_output);
        let hidden_errors = __Matrix.multiply(who_t, output_errors);

        //Calculate hidden gradient
        let hidden_gradient = __Matrix.map(hidden2, (x) => {
            return x * (1 - x);
        });
        hidden_gradient.multiply(hidden_errors);

        hidden_gradient.multiply(this.learning_rate);

        //Calculate input->hidden deltas
        let inputs_T = __Matrix.transpose(input);
        let weight_ih_deltas = __Matrix.multiply(hidden_gradient, inputs_T);

        //Adjust the weights by deltas
        this.weights_input_to_hidden.add(weight_ih_deltas);
        this.bias_hidden1.add(hidden_gradient);

        //Calculate the hidden->hidden deltas
        let hidden1_T = __Matrix.transpose(hidden1);
        let weight_hh_deltas = __Matrix.multiply(hidden_gradient, hidden1_T);

        //Adjust the weights by deltas
        this.weights_hidden_to_hidden.add(weight_hh_deltas);
        this.bias_hidden2.add(hidden_gradient);
    }

    async fit(x, y, settings={epochs: 100}) {
        const epochs = settings.epochs;
        let currentTime = new Date().getTime();
        let delta1000EpochTime = 0;
        let totalTime = 0;

        for (let i = 0; i < settings.epochs; i++) {
            let sampleN = Math.floor(Math.random() * x.length);
            await this.train(x[sampleN], y[sampleN]);

            if (i % 1000 == 0) {
                console.log("Current Epoch: " + i);
                if (i == 1000) {
                    console.log("Reached first 1000 epochs! Some data for the rest of the epochs:");
                    const deltaEpochTime = new Date().getTime() - currentTime;
                    delta1000EpochTime = deltaEpochTime;
                    console.log("Time for 1000 epochs: " + Math.floor(deltaEpochTime / 1000) + "s");
                    const timeTotal = deltaEpochTime * (epochs / 1000); //In ms
                    totalTime = timeTotal;
                    console.log("Expected time in total " + Math.floor(timeTotal / 1000) + "s");
                    console.log("Time left: " + Math.floor((timeTotal - deltaEpochTime) / 1000) + "s");
                }

                if (i % 10000 == 0 && i > 1000) {
                    console.log("% through epochs: " + Math.floor((i / epochs) * 100) + "%");
                    console.log("Time taken so far: " + Math.floor((delta1000EpochTime * (i / 1000)) / 1000) + "s");
                    console.log("Expected time in total " + Math.floor(totalTime/1000) + "s");
                    //time in ms - (timePer1000Epochs * time for 1000 epochs)
                    console.log("Time left: " + Math.floor((totalTime - (delta1000EpochTime * (i / 1000))) / 1000) + "s");
                }

                if (Object.keys(settings).includes("saving")) {
                    if (i % settings.saving.saveEvery == 0 && i > 1) {
                        let timestamp = new Date().getTime();
                        
                        this.save(settings.saving.pathFolder + "update-" + timestamp);

                        console.log("Saved to " + settings.saving.pathFolder + "update-" + timestamp + ".json");
                    }
                }
                
            }

        }
    }
}

if (usingNodeJS) {
    module.exports = { NeuralNetwork, DeeperNeuralNetwork };
}