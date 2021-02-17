# BayesianNN
In this work we introduce you to neural network and bayesian neural network. We explain briefly how they are build and how they work, then we inspect the strengths and weaknesses comparing the two approaches on at theoretical level and secondly applying these methods to a dataset.

## Structure

On the Networks folder 3 codes can be found:

ClassicalNeuralNetwork.py that implements a classical Neural Network.

HamiltonianMonteCarlo.py that implements a bayesian Neural Network with a Hamiltonian method for posteriori approximation.

VariationalInference.py that implements a bayesian Neural Network with a Variational Inference method for posteriori approximation.
## Dataset

It can be found in the following link:
```
https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
```
## External packages

Both Bayesian Neural Networks are implemented using Torch, in the HamiltonianMonteCarlo.py implementation we used the external package Hamiltorch and in the Variational Inference  Pyro. The classical Neural Network was implemented using Keras.

To install the packages you have to execute the following commands:

```
pip install keras
pip install pyro
pip install git+https://github.com/AdamCobb/hamiltorch
pip install pyro-ppl
```

## Build with

```
https://github.com/AdamCobb/hamiltorch
https://github.com/pytorch/pytorch
```
