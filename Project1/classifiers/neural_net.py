from __future__ import print_function

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
    # COMPLETE: Perform the forward pass, computing the class scores for the input. #
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
    activate_first_layer[activate_first_layer <= 0] = 0

    N, H = activate_first_layer.shape

    # In the same way, we append b2 to W2, and correct activate_first_layer
    
    addition_first_layer = np.ones((H, ))
    correction_second_input = np.concatenate((activate_first_layer, addition_first_layer.reshape(-1, 1)), axis = 1) # Stack column direction
    correction_second_weight = np.concatenate((W2, b2.reshape(1, -1)), axis = 0) # Stack row direction
    scores = np.matmul(correction_second_input, correction_second_weight) # (N, H+1) * (H+1, C) => (N, C)

    # Finally, we process softmax
    scores = np.exp(scores) / np.sum(np.exp(scores), axis = 0)
	
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
	
    # Softmax_error = -np.sum(np.log(np.exp(scores)/np.sum(np.exp(scores), axis = 0)) * y, axis = 0) / self.output_size or -np.sum(np.log(scores) * y, axis = 0)
    softmax_error = -np.log(scores)[np.argmax(y)] / self.output_size
    # Although summation is more compatiable(or generally, .. ), it is slower than using argmax.
    # More specifically, let's take a look following codes & result

    #############################################################################
    # 26.3 µs ± 613 ns per loop (mean ± std. dev. of 7 runs, 15000 loops each) / %timeit -n 15000 k = -np.sum(np.log(np.exp(a)/np.sum(np.exp(a), axis = 0)), axis = 0)
    # 24.9 µs ± 182 ns per loop (mean ± std. dev. of 7 runs, 15000 loops each) / %timeit -n 15000 k = -np.log(np.exp(a) / np.sum(np.exp(a), axis = 0))[np.argmax(ans)]
    # where a, and ans are vectors which both shapes are (1000, )
    # As aboves, we can know that using argmax is little faster (But enough!) as the expense of compactiable.
    #############################################################################

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

    losses = [softmax_error, regularization_w1, regularization_w2] 

    loss = np.sum(losses)

    #############################################################################
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
    pass
	
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
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
	  # - See [ np.random.choice ]											  #
      #########################################################################
	  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	  
      pass
	  
	  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################
	  
      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg) # loss function you completed above
      loss_history.append(loss)
	  
      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
	  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	  
      pass
	  
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
    # TODO: Implement this function; it should be VERY simple!                #
	# perform forward pass and return index of maximum scores				  #
    ###########################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
    pass
	
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


