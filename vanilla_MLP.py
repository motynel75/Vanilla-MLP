
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize

def LogisticSigmoid(X):
    """
    Computes the sigmoid activation function."""

    return 1.0/(1+ np.exp(-X))

def Softmax(Y):
    """
    Computes the softmax activation function for classification."""

    return np.exp(Y) / np.sum(np.exp(Y), axis=0)

def CrossEntropyLoss(Y, Y_true):
    """
    Computes the cross entropy loss function."""
    cost_sum = np.sum(np.multiply(Y_true, np.log(Y)))
    m = Y.shape[0]
    cost = -(1/m) * cost_sum

    return cost


def is_empty(alist):
    if alist:
        return False
    else:
        return True


class MultiLayerPeceptron:
    """creating Multi-layer perceptron (feedforward Neural network) model class."""
    def __init__(self, x_input, y_input, nb_class):
        self.input = x_input                  # Input data (matrix)
        self.y = y_input                      # Output data (vector)
        self.nb_class = nb_class              # Number of output class
        self.weights_vectors = []             # Creating all the wij weights matrices
        self.hidden_vectors = []              # Creating model's hidden vectors
        self.bias_vectors = []                # Creating model's bias vectors (associated to hidden vectors)

        self.momentum_weights_vectors = []
        self.momentum_bias_vectors = []

        self.output = np.zeros(self.nb_class) # Initial output vector (prediction)
        self.output_indicator = True          # To initialize output weights and bias
        self.momentum_indicator = True        # To initialize momentum term
        self.dropout_indicator = []           # Boolean vector to indicate dropout

    def addHiddenLayer(self, nb_neurons, dropout=0.):
        """Adding hidden layers to model by creating new weights matrices. Weights are
        randomly initialized."""

        # First hidden layer case
        if is_empty(self.weights_vectors):
            weights = np.random.rand(nb_neurons, self.input.shape[1])* np.sqrt(2. / self.input.shape[1] + nb_neurons) # Adjusting weights variance

        # All other hidden layer case
        else:
            weights = np.random.rand(nb_neurons, self.weights_vectors[-1].shape[0])*np.sqrt(2. / self.weights_vectors[-1].shape[0] + nb_neurons)

        hidden_vector = np.random.rand(nb_neurons) # Creates hidden layer (random activations)
        bias_vector = np.ones((nb_neurons,1))      # Creates bias vectors

        self.hidden_vectors.append(hidden_vector)
        self.bias_vectors.append(bias_vector)
        self.weights_vectors.append(weights)

        # Initialises dropout
        if dropout == 0. :
            # Default case (no dropout)
            self.dropout_indicator.append((False, 0.))
        else:
            self.dropout_indicator.append((True, dropout)) # Dropout case


        return self

    def feedforward(self, x_input):
        """Compute feedforward propagation from input data tensor x_input. Logistic sigmoid for hidden layers
        and softmax activation fuction for output layer are considered."""
        # Creates initial random weight for the output layers
        # Executes only once
        if self.output_indicator:
            weights = np.random.rand(self.output.shape[0], self.weights_vectors[-1].shape[0])* np.sqrt(2. / self.weights_vectors[-1].shape[0] + self.output.shape[0])
            bias_vector = np.ones((self.nb_class,1))

            self.bias_vectors.append(bias_vector)
            self.weights_vectors.append(weights)

            self.output_indicator = False

        x = x_input
        for nb_layers in range(len(self.hidden_vectors)):

            b = np.asarray(self.bias_vectors[nb_layers])          # Current bias
            W = np.asarray(self.weights_vectors[nb_layers])       # Current weights

            h_fwd = LogisticSigmoid(np.matmul(W,x.T) + b)         # Computes forward hidden layer activation

            #Add dropout
            if self.dropout_indicator[nb_layers][0]:                # Check if dropout is not null for current hidden layer
                dropout_rate = self.dropout_indicator[nb_layers][1] # Get dropout rate from tuple
                keep_prob = 1-dropout_rate
                d = np.random.rand(self.hidden_vectors[nb_layers].shape[0],1) # Create boolean with randomnly kept units
                d = d < keep_prob
                h_fwd = np.prod((h_fwd, d), axis=0)        # Prune activation neurons w.r.t dropout rate
                h_fwd = h_fwd/keep_prob                    # Scale the value of neurons that haven't been shut down

            h_fwd = h_fwd.T                                # re-setting dimesions

            #Updating activations
            self.hidden_vectors[nb_layers] = h_fwd        # Update object hidden layer attributes
            x = h_fwd

        # Updating output
        W_out = np.asarray(self.weights_vectors[-1])
        b_out = np.asarray(self.bias_vectors[-1])

        self.output = Softmax(np.matmul(W_out,x.T) + b_out)

        y_output = self.output

        return self, y_output

    def backpropagation(self, y_output, y_true, lr, beta, x_input):
        """Computes backpropagation from predicted output and true label.
        - lr   : represents the gradiant algorithm learning rate
        - beta : represents the momentum term coefficient."""

        m = y_output.shape[0] # get batch size

        y_error = np.asarray(y_output - y_true)

        ## Computes backpropagation for output layer (computed for Softmax)
        dW_out = (1./m)*np.matmul((y_error).T, self.hidden_vectors[-1])
        db_out = (1./m)*np.sum(y_error, axis=0, keepdims=True).T

        # initializes momentum terms (run once)
        if self.momentum_indicator:

            # Creating momentum terms vectors for hidden layers
            for nb_layers in range(len(self.hidden_vectors)):
                V_dW = np.zeros(self.weights_vectors[nb_layers].shape)
                V_db = np.zeros(self.bias_vectors[nb_layers].shape)

                self.momentum_weights_vectors.append(V_dW)
                self.momentum_bias_vectors.append(V_db)

            # Creating momentum terms vectors for output layer
            V_dW_out = np.zeros(self.weights_vectors[-1].shape)
            V_db_out = np.zeros(self.bias_vectors[-1].shape)

            self.momentum_weights_vectors.append(V_dW_out)
            self.momentum_bias_vectors.append(V_db_out)

            self.momentum_indicator = False

        # compute momentum terms for output
        V_dW_out = np.asarray(self.momentum_weights_vectors[-1])
        V_db_out = np.asarray(self.momentum_bias_vectors[-1])

        self.momentum_weights_vectors[-1] = (beta * V_dW_out + (1. - beta) * dW_out)
        self.momentum_bias_vectors[-1] = (beta * V_db_out  + (1. - beta) * db_out)

        # Update output parameter using momentum
        W_out = np.asarray(self.weights_vectors[-1])
        b_out = np.asarray(self.bias_vectors[-1])

        V_dW_out2 = np.asarray(self.momentum_weights_vectors[-1])
        V_db_out2 = np.asarray(self.momentum_bias_vectors[-1])

        self.weights_vectors[-1] = W_out - lr * V_dW_out2
        self.bias_vectors[-1]    = b_out - lr * V_db_out2

        y_error = y_error.T
        dh_L = y_error

        ## Computes backpropagation for all other layers (computed for Sigmoid)
        for nb_layers in reversed(range(len(self.hidden_vectors))):

            h = np.asarray(self.hidden_vectors[nb_layers].T)      # Forward hidden layer
            b = np.asarray(self.bias_vectors[nb_layers])          # Forward bias
            W = np.asarray(self.weights_vectors[nb_layers])       # Current layer weights
            W_L = np.asarray(self.weights_vectors[nb_layers+1])   # Forward layer weights

            dh = np.matmul(W_L.T,dh_L)* h * (1 - h)
            if nb_layers == 0:
                dW = (1./m) * np.matmul(dh, x_input)

            else:
                dW = (1./m) * np.matmul(dh, self.hidden_vectors[nb_layers-1])

            db = (1./m)*np.sum(dh, axis=1, keepdims=True)

            # compute momentum terms
            V_dW = np.asarray(self.momentum_weights_vectors[nb_layers])
            V_db = np.asarray(self.momentum_bias_vectors[nb_layers])

            self.momentum_weights_vectors[nb_layers] = (beta * V_dW + (1. - beta) * dW)
            self.momentum_bias_vectors[nb_layers] = (beta * V_db  + (1. - beta) * db)

            # Update parameter using momentum
            self.weights_vectors[nb_layers] = W - lr * V_dW
            self.bias_vectors[nb_layers]    = b - lr * V_db

            dh_L = dh

        return self

    def predict(self, X_test, Y_test):
        """Computes predicted output through feedforward method considering a test dataset."""

        predictions = []
        accurate = 0
        n = X_test.shape[0]

        # For all samples of X_test (no one-hot encode)
        for x in range(n):

            _, y_pred = self.feedforward(X_test[x].reshape(1,X_test.shape[1]))
            predictions.append(np.where(y_pred == np.amax(y_pred), 1., 0.).T[0])

        # Computes test accuracy
        for y in range(len(predictions)):

            if np.array_equal(predictions[y], Y_test[y]):

                accurate += 1
        accuracy = accurate / Y_test.shape[0]

        # calculer Accuracy, TPR, FPR, Precision, F1-score, recall ---> sickitlearn
        return predictions, accuracy

    def trainingProcess(self, nb_epoch, batch_size, lr, beta):
        """ Creates random mini-batch and apply training routine over each epoch
        (feedforward following by backpropagation for each batch). Output training, validation
        accuracy and loss function value over epochs. Prints loss function minimum and training
        accuracy graph."""

        m = self.input.shape[0]
        train_loss = []
        train_accuracies = []

        batches = -(-m // batch_size)
        print("nb of batches :", batches)

        X_train, X_val, Y_train, Y_val = train_test_split(self.input, self.y, test_size=0.20)

        for e in range(nb_epoch):

            # Creates random minibatch
            X_train_shuffled, Y_train_shuffled = shuffle(X_train, Y_train)

            average_cost = 0
            for b in range(batches):

                begin = b * batch_size
                end = min(begin + batch_size, X_train.shape[0]-1)

                X_batch = X_train_shuffled[begin:end]
                Y_batch = Y_train_shuffled[begin:end]

                if X_batch.shape[0] != 0: # Prevent empty batch

                    forward, y_output = self.feedforward(np.asarray(X_batch))
                    y_output = y_output.T

                    train_cost = CrossEntropyLoss(y_output, Y_batch)
                    average_cost += train_cost

                    backprob = self.backpropagation(y_output, Y_batch, lr, beta, np.asarray(X_batch))


            # Validation accuracy:
            pred_val, val_acc = self.predict(X_val, Y_val)

            # Train accuracy:
            pred, train_acc = self.predict(self.input, self.y)

            print(" * Epoch {}: Average cost = {}, Train_acc = {}, Val_acc = {}  * ".format(e+1, average_cost/batches, train_acc, val_acc))
            print("*******************************************************")

            # Store average training loss  and training accuracy value
            epoch_loss = average_cost/batches
            train_loss.append(epoch_loss)
            train_accuracies.append(train_acc)

        # Print loss and accuracy graph
        epoch = np.arange(0, nb_epoch, 1)
        plt.plot(epoch, train_loss)
        plt.savefig("loss.png")
        plt.xlabel('Epochs')
        plt.ylabel('Loss function minimum')
        plt.title('Train loss function w.r.t epochs')
        plt.show()

        plt.plot(epoch, train_accuracies)
        plt.savefig("acc.png")
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Train accuracy w.r.t epochs')
        plt.show()

        return self


# Loading dataset
iris = datasets.load_iris()
X_data = iris.data
Y_data = iris.target


# Shuffles dataset
X_data, Y_data = shuffle(X_data, Y_data)

#Noramlizes the data
X_normalized = normalize(X_data,axis=0)
X_data = X_normalized

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.20)


# Test and train one-hot encoding

# integer encode
label_encoder = LabelEncoder()

Y_train_enc = label_encoder.fit_transform(Y_train)
Y_test_enc = label_encoder.fit_transform(Y_test)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)

Y_train_enc = Y_train_enc.reshape(len(Y_train_enc), 1)
Y_test_enc = Y_test_enc.reshape(len(Y_test_enc), 1)

Y_train_enc = onehot_encoder.fit_transform(Y_train_enc)
Y_test_enc = onehot_encoder.fit_transform(Y_test_enc)


# Main :
if __name__ == '__main__':

    # Creates empty feedforward network
    model = MultiLayerPeceptron(X_train, Y_train_enc, nb_class=3)

    # Adding hidden layers
    model = model.addHiddenLayer(64, dropout=0.)

    # Running training routine
    model = model.trainingProcess(nb_epoch=500, batch_size=1, lr=.04, beta=0.9)

    # Prediction on test dataset:
    pred, test_acc = model.predict(X_test, Y_test_enc)
    print("*** Test Accuracy ***: {}".format(test_acc))
