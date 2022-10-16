from __future__ import print_function
from turtle import shape

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
	# np.random.randn(shape)
	# - Return a sample (or samples) from the “standard normal” distribution following shape
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
    # Note that shape of W1: (Input_size, hidden_size) / b1: (hidden size, )
    # As amount of sample is N, input's dimension is D
    # At first, let's calculate first layer!
    # For convinence, let's append b1 to W1, and correct at X

    addition_input = np.ones((N, ))
    correction_input = np.concatenate((X, addition_input.reshape(-1, 1)), axis = 1) # Stack column direction
    correction_first_weight = np.concatenate((W1, b1.reshape(1, -1)), axis = 0) # Stack row direction
    not_activate_first_layer = np.matmul(correction_input, correction_first_weight) # (N, D+1) * (D+1, H) => (N, H)
    activate_first_layer = not_activate_first_layer.copy()
    activate_first_layer[activate_first_layer < 0] = 0
    
    N, H = activate_first_layer.shape

    # In the same way, we append b2 to W2, and correct activate_first_layer
    
    addition_first_layer = np.ones((N, ))
    correction_second_input = np.concatenate((activate_first_layer, addition_first_layer.reshape(-1, 1)), axis = 1) # Stack column direction
    correction_second_weight = np.concatenate((W2, b2.reshape(1, -1)), axis = 0) # Stack row direction
    scores = np.matmul(correction_second_input, correction_second_weight) # (N, H+1) * (H+1, C) => (N, C)

    N, C = scores.shape

    # Finally, we process softmax
    # print(scores_bef)
    # print(np.sum(np.exp(scores_bef), axis = 1).reshape(-1, 1))
    

	
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # COMPLETE: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
    # First, we need to calculate loss per data i, and then summation of that
    # Apply fancy indexing with ground truth's transpose, and then np.sum
    # Shape of scores[N, y] is (N, ), so apply np.sum with axis = 0 (not necessary, but it is better to explict information of axis)
    scores = scores - np.max(scores, axis = 1).reshape(-1, 1)
    scores = np.exp(scores) / np.sum(np.exp(scores), axis = 1).reshape(-1, 1)
    scores_after = np.log(scores)
    softmax_error = - np.sum(scores_after[np.arange(N), y], axis = 0) / N

    # L2 regularizaiton for W1 / parameter : reg
    # Surely, W1 ** 2 is easy to modify. But previous code is slower than W1 * W1! (Why?)

    #############################################################################
    # 1.41 µs ± 493 ns per loop (mean ± std. dev. of 7 runs, 15000 loops each) / %timeit -n 15000 k**2
    # 1.08 µs ± 6.39 ns per loop (mean ± std. dev. of 7 runs, 15000 loops each) / %timeit -n 15000 k*k
    # where shape of k is (30, 30)
    # As aboves, we can know that using * is little faster (very very little) as disadvantage of modifying.
    #############################################################################

    # ** is wrapper for the pow function, but * is arithmetic operator, so ** is more overhead than *
    # Are square, cube, ... and more conserved?
    # Hence divide & conquer mechanism works on pow function, I think that isn't conserved... (If use dynamic progamming with operator *, that law will be conserved!)

    regularization_w1 = reg * np.sum(W1 * W1)

    # In the same way, we can calculate regularization about W2

    regularization_w2 = reg * np.sum(W2 * W2)

    # If we modify(i.e. add, subtract, ...) some loss function, we can deal with easily using append() function.

    losses = np.array([softmax_error, regularization_w1, regularization_w2])


    loss = np.sum(losses)

    #############################################################################
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # FIXME: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Structure of grads: {'W1': ~, 'b1': ~, 'W2': ~, 'b2': ~}
    # We will calculatr gradients of aboves based on back propagation, which reuse previous layer's information (Not only gradients but kind of activation, loss, and so on)
    # Loss function is scalar, probability with softmax is (N, ), Result is (N, C) (Before softmax)
    # MEMO: Think Expend dimension to parallize input datas? => Not necessary..? => einsum after summation
    
    # First, we will get gradient about probability (furthermore, output of second layer)
    # To caclculate, we will pre-calculate padding of y (y: (N, 1) => (N, C))
    back_y = np.zeros((N, C))
    back_y[np.arange(N), y] = 1
    # print(scores, back_y, scores - back_y, sep = '\n')
    # print("==========================================")
    output2_grad = (scores - back_y) / N # scores : (N, C), back_y : (N, C) / Element-wise operation. => output2_grad : (N, C)

    # Let us calculate gradient of W2, b2!
    # Note that Z_2 = W_2 x + b
    # Therefore gradient of local and global are (dz / dw = )x^T, (dL / dz = )output2_grad * x^T, respectively.
    # Hence weight will be calculated each dataset, we have to expand dimension like (N, C, 1) and (N, H, 1), then perform matrix multiplication (i.e. outer product of two matrix)
    # For convenience, we will use 'np.einsum' to calculate simply, and then summation by axis-0 to use np.sum ((N, H, C) -> (H, C))
    # Surely, we can calculate using matrix multiplication (1. expand output2_grad and activate_first_layer to use np.expand, then calculate)
    # More specifially, weight2_grad = np.sum(np.expand_dims(activate_first_layer, axis = 2) @ np.expand_dims(output_2grad, axis = 1), axis = 0)
    weight2_grad = np.sum(np.einsum('ij,ik -> ijk', activate_first_layer, output2_grad), axis = 0) # activate_first_layer : Input of second layer
    z2_grad = np.matmul(output2_grad, W2.T) # output2_grad: (N, C) / W2: (H, C) => W2.T: (C, H) / z2_grad: (N, H)
    
    # In the same way, let's calculate the bias.
    # As not only shape of bias is (C, ) but local gradient of bias consists only 1, gradient of bias with respective to loss is same as output2_grad(dL / dz)
    bias2_grad = np.sum(output2_grad, axis = 0)

    # Insert into dictionary, called 'params'
    grads['W2'] = weight2_grad + 2 * reg * W2
    grads['b2'] = bias2_grad

    # Now we have to calculate (global) gradient of x2(i.e. activate_first_layer)
    # That will be diagonal matrix, which element of diagonal is composition of activation function and z1
    # We use relu by activation function, detrivate cofficient is 1 if input is positive, 0 otherwise.

    # MEMO: Please dobule-check following code & formula.
    activate_first_layer_grad = np.zeros(shape = (N, H, H))
    diag_check = np.where(not_activate_first_layer >= 0) # Extracting only positive values' indices.
    activate_first_layer_grad[diag_check[0], diag_check[1], diag_check[1]] = 1 # Size : (N, H, H)
    
    # output1_grad = activate_first_layer_grad * np.expand_dims(z2_grad, axis = -1)
    
    output1_grad = np.einsum('ijk, ik -> ij', activate_first_layer_grad, z2_grad) # Size : (N, H)
    weight1_grad = np.sum(np.einsum('ij, ik -> ijk', X, output1_grad), axis = 0) # Size: (N, D) * (N, H) ->(einsum) (N, D, H) ->(sum) (D, H)

    bias1_grad = np.sum(output1_grad, axis = 0)

    # Insert into dictionary, called 'params'
    grads['W1'] = weight1_grad + 2 * reg * W1
    grads['b1'] = bias1_grad

	
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # COMPLETE: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
	  # - See [ np.random.choice ]											  #
      #########################################################################
	  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	  
      indices = np.random.choice(a = X.shape[0], size = batch_size)

      X_batch, y_batch = X[indices], y[indices]

	  
	  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################
	  
      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg) # loss function you completed above
      loss_history.append(loss)
	  
      #########################################################################
      # COMPLETE: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
	  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	  
      self.params['W1'] = self.params['W1'] - learning_rate * grads['W1']
      self.params['W2'] = self.params['W2'] - learning_rate * grads['W2']
      self.params['b1'] = self.params['b1'] - learning_rate * grads['b1']
      self.params['b2'] = self.params['b2'] - learning_rate * grads['b2']
	  
	  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

	  # print loss value per 100 epoch
      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # COMPLETE: Implement this function; it should be VERY simple!                #
	# perform forward pass and return index of maximum scores				  #
    ###########################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
    # print(np.matmul(X, self.params['W1']).shape, self.params['b1'].shape, sep='\n')
    First_layer = np.matmul(X, self.params['W1']) + self.params['b1']
    First_layer[First_layer < 0] = 0

    Second_layer = np.matmul(First_layer, self.params['W2']) + self.params['b2']

    y_pred = np.argmax(Second_layer, axis = 1)
	
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


