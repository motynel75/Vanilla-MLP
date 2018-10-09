# Multilayer perceptron

This project shows my own implementation of a feedforward neural network using Python 3 Numpy library. The model was tested on the IRIS dataset, containing 150 samples of 3 class each, to classify plant species. This version includes Logistic sigmoid activation function for hidden layers, Softmax function for output layer and Cross-entropy as loss function. We achieved between 91.3% and 100% of accuracy on this dataset.

This implementation includes some basic features:

- Stochastic gradient descent
- Momentum
- Dropout
- Xavier initializer

More improvement techniques will be add in further versions.

### Prerequisites

Use your Python 3 Numpy version.

Use sickit-learning package for dataset processing.
```
pip install -U scikit-learn
```

### Installing and runining

In order to use this model, you can directly compile either the providedvanilla_MLP.pyorvanilla_mlp.ipynb. The code already contains a model with the optimal set of hyper-parameters. If you need to create your own configuration, you have to set up a model as following :

To create a model:
```
model = MultiLayerPeceptron(X_train, Y_train_enc, nb_class=NUMBER_OF_CLASS)
```
Add a number of hidden layers:
```
# Adding hidden layers
model = model.addHiddenLayer(NUMBER_OF_NEURONS, dropout=0.)
model = model.addHiddenLayer(NUMBER_OF_NEURONS, dropout=0.)
model = model.addHiddenLayer(NUMBER_OF_NEURONS, dropout=0.)
...

```

Run training routine:
```
# Running training routine
model = model.trainingProcess(nb_epoch=NB_EPOCHS, batch_size=1, lr=LEARNING_RATE, beta=0.9)

```

Compute test predictions and accuracy.
```
# Prediction on test dataset:
pred, test_acc = model.predict(X_test, Y_test_enc)
print("*** Test Accuracy ***: {}".format(test_acc))

```

## Authors

* **Amine K.** - *Initial work* - [Vanilla MLP](https://github.com/motynel75/Vanilla-MLP)
