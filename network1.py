import numpy as np
import nnfs
import matplotlib.pyplot as plt

from nnfs.datasets import spiral_data



nnfs.init()

X = [[3.3, 4.4 , 55, 2.5], 
     [2.0, 5.0, -1.0, 2.0], 
     [1.0, 2.0, 3.0, 9.8]]

X, y = spiral_data(100, 3)


class Layer_Dense:
        #Layer Init
        def __init__ (self, inputs, neurons):
            self.weights = 0.01 * np.random.randn(inputs, neurons)
            self.biases = np.zeros((1, neurons))
            
        #Forward Pass
        def forward(self, inputs):
            
            self.inputs = inputs
            self.output = np.dot(inputs, self.weights) + self.biases
        
        #Backward Pass 
        def backward(self, dvalues):
            
            #partial der. w.r.t. weights
            self.dweights = np.dot(self.inputs.T, dvalues)
            
            #partial der. w.r.t. biases
            self.dbiases = np.sum(dvalues, axis=0, keepdims = True)
            
            #partial der. w.r.t. inputs
            self.dinputs = np.dot(dvalues, self.weights.T)

            
class Activation_ReLU:
    
    #forward pass (receives input from corresponding summation and bias of a layer)
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    #backward pass    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        
        #gradient is zero if the input values were negative
        self.dinputs[self.inputs <= 0] = 0
        
class Activation_Softmax:
    
    #forward pass
    def forward(self, inputs):
        
        self.inputs = inputs
        
        #subtracting every value by the maximum value in the sample (row), 
        #does not affect the normalization but prevents values from blowing up
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        
        #normalize the values (0-1)
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities
    
    #backward pass
    def backward(self, dvalues):
    
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
                    
            single_output = single_output.reshape(-1, 1)
            
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
                              
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

#Parent Loss Class          
class Loss:
    
    #compares model output to ideal output to calculate loss
    def calculate(self, output, y):
        
        sample_losses = self.forward(output, y)
        
        data_loss = np.mean(sample_losses)
        
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    
    #Forward Pass
    def forward(self, y_pred, y_true):
        
        #y_pred is a matrix so 'len' returns # of rows (samples)
        samples = len(y_pred)
        
        #prevent divide-by-zero errors
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        #Summation?
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
            
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
        
        #Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    #Backward Pass
    def backward(self, dvalues, y_true):
        
        #Number of samples
        samples = len(dvalues)
        
        #Number of values per sample
        labels = len(dvalues[0])
        
        #if the true values are in list form
        if len(y_true.shape) == 1:
            
            #creates an identity matrix and accesses the appropriate row vector
            y_true = np.eye(labels)[y_true]
        
        #calculate gradient    
        self.dinputs = -y_true / dvalues
        
        #normalize gradient
        self.dinputs = self.dinputs / samples
        
#combines backward pass for Loss function & Sigmoid Activation function
class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    #initialize instances of each function
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
     
    #combined forward pass   
    def forward(self, inputs, y_true):
        
        #calculate last layer's output
        self.activation.forward(inputs)
        self.output = self.activation.output
        
        #return the loss of that output
        return self.loss.calculate(self.output, y_true)

    #backward pass
    def backward(self, dvalues, y_true):
        
        #Number of samples
        samples = len(dvalues)
        
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
            
        self.dinputs = dvalues.copy()
        
        self.dinputs[range(samples), y_true] -= 1
        
        self.dinputs = self.dinputs / samples
        
class Optimizer_SGD:
    
    def __init__(self, learning_rate = 1.0, decay = 0.0, momentum = 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
        
    def update_params(self, layer):

        if self.momentum:
            
            if not hasattr(layer, 'weights_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates    
                
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        layer.weights += weight_updates
        layer.biases += bias_updates
            
        
    def post_update_params(self):
        self.iterations += 1    
        
class Optimizer_RMSProp:
    
    def __init__(self, learning_rate = 1e-3, decay = 0.0, epsilon = 1e-7, rho = 0.9):
        
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))     
    
    def update_params(self,layer):
        
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2
            
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
            
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
                (np.sqrt(layer.bias_cache) + self.epsilon)
        
    def post_update_params(self):
        self.iterations += 1
        
    
class Optimizer_SGD:
    
    def __init__(self, learning_rate = 1.0, decay = 0.0, momentum = 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
        
    def update_params(self, layer):

        if self.momentum:
            
            if not hasattr(layer, 'weights_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates    
                
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        layer.weights += weight_updates
        layer.biases += bias_updates
            
        
    def post_update_params(self):
        self.iterations += 1    
        
class Optimizer_Adam:
    
    def __init__(self, learning_rate = 1e-3, decay = 0.0, epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))     
    
    def update_params(self,layer):
        
        if not hasattr(layer, 'weight_cache'):
            
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + \
                (1 - self.beta_1) * layer.dweights
                
        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + \
                (1- self.beta_1) * layer.dbiases
                
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
            
            
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
            
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
            
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
            
        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
            
        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    def post_update_params(self):
        self.iterations += 1
                            
        
#initialize a spiral data set with 3 spirals
X, y = spiral_data(samples = 100, classes = 3)

print(X)
#create dense layer with 2 inputs (x,y coords) and 3 neurons
dense1 = Layer_Dense(2, 64)

#activation function for the first layer
activation1 = Activation_ReLU()

#second dense layer must have num_inputs = num_outputs from prev layer
dense2 = Layer_Dense(64, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#initialize an optimizer
optimizer = Optimizer_Adam(learning_rate = 0.01, decay = 1e-5)

#track loss and accuracy to generate plots
losses = []
accuracies = []

for epoch in range(10001):
    
    #step 1: perform a forward pass
    dense1.forward(X)
    
    activation1.forward(dense1.output)
    
    dense2.forward(activation1.output)
    
    #this line combines sigmoid activation and loss
    loss = loss_activation.forward(dense2.output, y)
    losses.append(loss)
    
    #print out the accuracy and loss of each epoch (each loop)
    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis = 1)
    
    accuracy = np.mean(predictions == y)
    accuracies.append(accuracy)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}')
        
    #step 2: perform a backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    #step 3: update parameters based on gradient from backward pass and decay learning rate
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
