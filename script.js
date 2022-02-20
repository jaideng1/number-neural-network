let nn;

let numberNeuralNetwork;

let grid = {
    width: 28,
    height: 28,
    scalar: 20,
    matrix: null,
}

function setup() {
    let c = createCanvas(grid.width * grid.scalar, grid.height * grid.scalar);

    const cc = document.querySelector("#canvas-container")
    document.querySelector("#cc").appendChild(c.canvas);
    // cc.innerHTML += clearButton;
    // cc.innerHTML += predictButton;

    nn = new NeuralNetwork(2, 10, 1);

    let inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ];

    let targets = [
        [0],
        [1],
        [1],
        [0]
    ];

    nn.fit(inputs, targets, 50000);

    grid.matrix = new Matrix(grid.width, grid.height);
    grid.matrix.initializeZeros();

    numberNeuralNetwork = new DeeperNeuralNetwork(784, 100, 10);

    numberNeuralNetwork.loadURL("/training-data/1645320483198_data/finished.json");
}

function getClosestGridPoint(x, y) {
    let closestX = Math.floor(x / grid.scalar);
    let closestY = Math.floor(y / grid.scalar);

    return {closestX, closestY};
}

function draw() {
    background(0,0,0);
    
    for (let x = 0; x < grid.width; x++) {
        for (let y = 0; y < grid.height; y++) {
            let val = grid.matrix.get(x, y);

            noStroke();

            fill(val * 255);
            rect(x * grid.scalar, y * grid.scalar, grid.scalar, grid.scalar);
        }
    }

    stroke(0,255,0);
    strokeWeight(1);
    noFill();
    if (mouseDown && !predictionOpen) {
        fill(0,255,0)

        let {closestX, closestY} = getClosestGridPoint(mouseX, mouseY);
        try {
            grid.matrix.set(closestX, closestY, 1);

            try {
                grid.matrix.set(closestX - 1, closestY, grid.matrix.get(closestX - 1, closestY) + 0.5);
                grid.matrix.set(closestX, closestY - 1, grid.matrix.get(closestX - 1, closestY) + 0.5);
            } catch (e) {}

            try {
                grid.matrix.set(closestX + 1, closestY, grid.matrix.get(closestX + 1, closestY) + 0.5);
                grid.matrix.set(closestX, closestY + 1, grid.matrix.get(closestX + 1, closestY) + 0.5);
            } catch (e) {}
        } catch (e) {}
    }

    ellipse(mouseX, mouseY, 10, 10);
}

function clearMatrix() {
    try {
        grid.matrix.initializeZeros();
    } catch (e) {
        grid.matrix = new Matrix(grid.width, grid.height);
        grid.matrix.initializeZeros();
    }
}

let predictionOpen = true;

function predictCanvas() {
    let inputs = [];
    for (let x = 0; x < grid.width; x++) {
        for (let y = 0; y < grid.height; y++) {
            inputs.push(grid.matrix.get(x, y));
        }
    }

    let prediction = numberNeuralNetwork.predict(inputs);

    let predictionKeys = {};

    for (let i = 0; i < prediction.length; i++) {
        predictionKeys[i] = prediction[i];
    }

    let sortedPredictionKeys = Object.keys(predictionKeys).sort(function(a, b) {
        return predictionKeys[b] - predictionKeys[a];
    });

    let predictionString = `<h3>I predict that your number is <b>${sortedPredictionKeys[0]}${predictionKeys[sortedPredictionKeys[0]] >= 0.5 ? "!" : "??"}</b></h3><br/>`;

    for (let i = 0; i < sortedPredictionKeys.length; i++) {
        predictionString += `${sortedPredictionKeys[i]}: ${predictionKeys[sortedPredictionKeys[i]]}<br>`;
    }

    document.getElementById("prediction").innerHTML = predictionString;

    document.getElementById("prediction-container").classList.remove("hidden");

    predictionOpen = true;
}

function closePrediction() {
    predictionOpen = false;
    document.getElementById("prediction-container").classList.add("hidden");
}

function exitExplination() {
    predictionOpen = false;

    document.getElementById("explination-container").classList.add("hidden");
}


let mouseDown = false;

function mousePressed() {
    mouseDown = true;
}

function mouseReleased() {
    mouseDown = false;
}