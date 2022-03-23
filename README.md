# number-neural-network
A small neural network I (tried) making to recognize numbers from 0 to 9 using the MNIST database.

View at https://jaideng1.github.io/number-neural-network/.  

This is a small neural network that I (tried) to make using the [MNIST DATABASE](http://yann.lecun.com/exdb/mnist/).  
  
Unfortunately, it's not very accurate. I only trained it for an hour though, so it's not too accurate.     
I would place the blame on:  
1. Accidentally repeating the same training data over and over again.
2. Not training it for long enough.
3. Accidentally reading the data wrong, either in training or testing.  
  
Play around with it though! (It seems to love the number 4).  

NOTE: I learned that I completely did the derivative of sigmoid wrong (augh). Turns out instead of `s'(x) = 1 * (1 - x)` it's actually `s'(x) = s(x) * (s(x) - x)`. I haven't changed this yet, but I will soon.

# Running this locally

This is also able to run locally with NodeJS for training.   
To do so, download the repository and extract it from the zip file, then create a folder called `training-data`.   
You'll have to change the training.js though in order for it to actually run.  
You also need to put the MNIST `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz` inside a folder named `mnist` inside the `training-data` folder.
I would replace the `testAI(100);` with `trainAI()` in order to do so. You can also change the `epochs` constant around line 78.  
  
If you want to use a pre-existing training file (created using this program), load it after the `let nn = new DeeperNeuralNetwork()` around line 64.  
Replace it with `nn.loadFile(<fileLocation relative to training.js's directory>)`. 

To run it, run `node training.js` in Terminal/CMD.  
  
(Sorry for this bad instructions lol)
