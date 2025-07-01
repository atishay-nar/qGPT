# qGPT

Pennylane and Pytorch implementation of QMSAN as part of an image GPT

qmsan.mathematical is a mathematical simulation of embedding the input vector to a the mixed state and computing swap test. running each pair through the swap circuit takes too much time on the simulator, so this is a more effective method of training. 

qmsan now does not project q and k classically and instead used an amplitude embedding

Seriously reducing the size of the MNIST data set to be able to train the model in a reasonable amount of time

QMSAN inspired by: https://arxiv.org/abs/2403.02871

Model inspired by: https://github.com/teddykoker/image-gpt

