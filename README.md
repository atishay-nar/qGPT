# qGPT

Pennylane and Pytorch implementation of QMSAN and QKSAN as part of an image GPT

mathematical.py is a mathematical simulation of embedding the input vector to a the mixed state and computing the results of a SWAP test. running each pair through the SWAP circuit takes too much time on the simulator, so this is a more effective method of training. circuit.py stays true to the nature of QMSAN and simulates O(S^2/2) SWAPs. This method does not feature a trainable embedding due to constraints with gradients and mixed states

trainable.py trades avoids mixed states so that gradients can be calculated and the quantum circuit can have parameterized gates. This is similar to a QKSAN, and I calculate attnetion scores with the JS divergence of the probability vectors for each q k pair that has run through the circuit. Gradient calculation greatly increases overhead and takes a ridiculous amount of time to train.

Seriously reducing the size of the MNIST data allowed model to train in a reasonable amount of time.

QMSAN inspired by: https://arxiv.org/abs/2403.02871

Model inspired by: https://github.com/teddykoker/image-gpt

