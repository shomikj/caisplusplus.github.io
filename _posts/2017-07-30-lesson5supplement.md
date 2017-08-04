---
layout: post
published: true
title: "Lesson 5 Supplemental Material"
headline: ""
mathjax: true
featured: false
categories: curriculum-supplement

comments: false
---

### Table of Contents

1. [Backpropagation](#backpropagation)
2. [Loss Functions](#loss-functions)
3. [Regularization](#regularization)
4. [Optimization in Practice](#optimization-in-practice)
5. [The Unstable Gradient Problem](#unstable-gradient)

<h2><a name="backpropagation"></a>Backpropagation</h2>

<p>
  In this section, we will generalize the results from the [one-layer training case](../curriculum-supplement/lesson4supplement) to
  any number of layers in our network and any activation functions. This will
  form the backpropagation algorithm which is used to train the parameters of a
  neural network.
</p>

<p>
  With our previous network, we were able to show that a single neuron network
  creates a linear decision boundary. This is certainly a powerful tool and
  capable of solving many problems, but obviously is incapable of solving
  non-linear functions. For instance, say we wanted to create a neural network
  to learn the XOR function. Let's plot the results of the XOR function (\( x_1
  \text{ XOR } x_2 \)) where the black dots are false and the gray are true.
</p>

<img class='center-image' src='/assets/img/ml/crash_course/xor.png' />

<p>
  Clearly there is no linear decision function that can be drawn to
  differentiate true from false. Even if the number of neurons in the layer was
  increased no single layer network could ever model this relationship.
</p>

<p>
  We need to use a two layer network for this task. In fact it can be proved
  that a two layer network has the capability of approximating any function,
  linear or non-linear. (However, this is not really practical as at a certain
  point the number of neurons would have to be increased dramatically. Deeper
  neural networks are far more advantageous and require less neurons).
</p>

<p>
  By the same logic as with the single layer we have the following where \( m
  \) is the layer index. Except this time calculating the derivatives is more
  complex. 

  $$ w_{i,j}^{m}(k+1) = w_{i,j}^{m}(k) - \alpha \frac{ \partial J}{\partial
  w_{i,j}^{m}} $$

  $$ b_{i}^{m}(k+1) = b_{i}^{m}(k) - \alpha \frac{ \partial J}{\partial
  b_{i}^{m}} $$
</p>

<p>
  Let's go ahead and employ the chain rule of calculus. Remember that 
  ignoring the bias, the product of the input and the weights is given by 
  \( n_{i}^{m} = w_{i}^{m} * p_{i}^{m} \). 
  This can be written as a function of the weights of the layer.
  Let's apply the chain rule knowing that \( n_{i}^{m} \) can be written as \(
  n_{i}^{m} ( w_{i,j} ) \). 

  $$ 
  \frac{ \partial J}{\partial w_{i,j}^{m}} = \frac{\partial J}{\partial
  n_{i}^{m}} * \frac{\partial n_{i}^{m}}{\partial w_{i,j}^{m}}
  $$

  $$ 
  \frac{ \partial J}{\partial b_{i}^{m}} = \frac{\partial J}{\partial n_{i}^{m}} *
  \frac{\partial n_{i}^{m}}{\partial b_{i}^{m}} $$
</p>

<p>
  Calculating the derivative of \( n_{i}^{m} \) is straightforward enough as we
  can write an explicit expression for \( n_{i}^{m} \).
  $$ n_{i}^{m}(w_{i,j}) = \sum_{j=1}^{S^{m-1}} w_{i,j}^{m} a_{j}^{m-1} + b_{i}^{m} $$

  Then take the derivative of both sides.

  $$ \frac{ \partial n_{i}^{m}}{\partial w_{i,j}^{m}}  = a_{j}^{m-1} $$
  $$ \frac{ \partial n_{i}^{m}}{\partial b_{i}^{m}} = 1 $$
</p>

<p>
  Now let's assign \( \frac{\partial J}{\partial n_{i}^{m}} \) to the arbitrary
  value \( s_{i}^{m} \) which we still do not know. We will call this term the
  sensitivity.
  $$ \frac{\partial J}{\partial n_{i}^{m}} = s_{i}^{m} $$
</p>

<p>
  So now we are left with the following.
  $$ w_{i,j}^{m}(k+1) = w_{i,j}^{m}(k) - \alpha s_{i}^{m} a_{j}^{m-1}$$

  $$ b_{i}^{m}(k+1) = b_{i}^{m}(k) - \alpha s_{i}^{m} $$

  Which in matrix form becomes: 
  $$ \textbf{W}^{m}(k+1) = \textbf{W}^{m}(k) - \alpha \textbf{s}^{m}
  (\textbf{a}^{m-1})^{T} $$

  $$ \textbf{b}^{m}(k+1) = \textbf{b}^{m}(k) - \alpha \textbf{s}^{m} $$

Where we have:
  $$\textbf{s}^{m} = \frac{\partial J}{\partial n^{m}} = 
  \begin{bmatrix}
    \frac{\partial J}{\partial n_{1}^{m}} \\
    \frac{\partial J}{\partial n_{2}^{m}} \\
    \vdots \\
    \frac{\partial J}{\partial n_{S^{m}}^{m}} \\
  \end{bmatrix} $$
</p>

<p>
  The goal of backpropagation equations is to come up with an expression for 
  \( \textbf{s}^{m} \) in terms of \( \textbf{s}^{m-1} \). To get to our result
  we first have to do some math magic so bear with me. 
</p>

<p>
  Consider the following matrix: (This is called a Jacobian matrix)
  $$
  \frac{\partial \textbf{n}^{m+1}}{\partial \textbf{n}^{m}} = 
  \begin{bmatrix}
    \frac{\partial n_{1}^{m+1}}{\partial n_{1}^{m}} &&
    \frac{\partial n_{1}^{m+1}}{\partial n_{2}^{m}} &&
    \dots &&
    \frac{\partial n_{1}^{m+1}}{\partial n_{S^{m}}^{m}} \\

    \frac{\partial n_{2}^{m+1}}{\partial n_{1}^{m}} &&
    \frac{\partial n_{2}^{m+1}}{\partial n_{2}^{m}} &&
    \dots &&
    \frac{\partial n_{2}^{m+1}}{\partial n_{S^{m}}^{m}} \\

    \vdots && 
    \vdots && 
    &&
    \vdots \\

    \frac{\partial n_{S^{m+1}}^{m+1}}{\partial n_{1}^{m}} &&
    \frac{\partial n_{S^{m+1}}^{m+1}}{\partial n_{2}^{m}} &&
    \dots &&
    \frac{\partial n_{S^{m+1}}^{m+1}}{\partial n_{S^{m}}^{m}} \\
  \end{bmatrix}
  $$

  It may not immediately be clear why we would need to bring this matrix into
  things but we will see that we can use it to find an expression for \( s^m
  \). Let's look at some arbitrary element in this matrix and see if we can simplify
  it.
  $$
    \frac{\partial n_{i}^{m+1}}{\partial n_{j}^{m}} = \frac{ \partial \left( \sum_{l=1}^{S^{m}} 
    w_{i,l}^{m + 1} a_{j}^{m} + b_{i}^{m + 1} \right) }{\partial n_{j}^{m}} =
    \frac{\partial (w_{i,j}^{m} * a_{j}^{m}(n_{j}^{m}))}{\partial n_{j}^{m}} = w_{i,j}^{m} \frac{\partial
    a_{j}^{m}(n_{j}^{m})}{\partial n_{j}^{m}}
  $$ 

  But we know the explicit formula for \( a_{j}^{m} \) ! 
  $$ a_{j}^{m} (n_{j}^{m}) = f^{m}(n_{j}^{m})$$

  Where \( f^{m} \) is the activation function for layer \( m \).

  $$
  \frac{\partial n_{i}^{m+1}}{\partial n_{j}^{m}} = w_{i,j}^{m} \frac{\partial
    f^{m}(n_{j}^{m})}{\partial n_{j}^{m}} = w_{i,j}^{m} D^m(n_{j}^{m})
  $$
  
  Where we have \( \frac{\partial f^{m}(n_{j}^{m})}{\partial n_{j}^{m}} =
  D^m(n_{j}^{m}) \) just to make the notation look easier. Note that \(
  d^m(n_j^m) \) is evaluating the derivative of \( f^m \) for the value \(
  n_j^m \). Doing this is easy for computers. 
</p>

<p>
  Now we have an explicit expression for \( \frac{\partial
  n_{i}^{m+1}}{\partial n_{j}^{m}} \) 

  $$ 
  \frac{\partial \textbf{n}^{m+1}}{\partial \textbf{n}^{m}} = \textbf{W}^{m +
  1} \textbf{D}^m (\textbf{n}^{m})
  $$
</p>

<p>
  Simplify the original sensitivity term.
  $$ 
  \textbf{s}^{m} = \frac{\partial J}{\partial \textbf{n}^{m}} = 
  \left( \frac{\partial \textbf{n}^{m+1}}{\partial \textbf{n}^{m}} \right)^{T}
  \frac{\partial J}{\partial \textbf{n}^{m + 1}}
  $$
  $$
  \textbf{s}^{m} = 
  \left( \textbf{W}^{m + 1} \textbf{D}^m (\textbf{n}^{m}) \right)^{T} 
  \frac{\partial J}{\partial \textbf{n}^{m + 1}} 
  $$
  $$
  \textbf{s}^{m} = 
  \textbf{D}^m (\textbf{n}^{m}) \left( \textbf{W}^{m + 1} \right)^{T}
  \frac{\partial J}{\partial \textbf{n}^{m + 1}} 
  $$
  $$
  \textbf{s}^{m} = 
  \textbf{D}^m (\textbf{n}^{m}) \left( \textbf{W}^{m + 1} \right)^{T}
  \textbf{s}^{m+1}
  $$

  From this expression we can see that \( \textbf{s}^m \) depends on \(
  \textbf{s}^{m+1} \) and so on. So we are sending the sensitivites backwards
  in the network. This is why it is called <b>backpropogation</b>. 
</p>

<p>
  What does the last layer depend on? ( \(\textbf{s}^{M} \)?)
  We can just consider this to be the base case for our recursive relation
  where the backpropogation starts.
  $$
  s_{i}^{M} = \frac{\partial J}{\partial n_{i}^{M}} = 
  \frac{ \partial \left( \sum_{j=1}^{S^M} (t_j - a_j)^2 \right)}{\partial n_{i}^{M}} 
  =
  -2(t_j-a_j)\frac{\partial a_i}{\partial n_{i}^{M}}
  $$
  $$
  s_{i}^{M} = -2(t_j-a_j)\frac{\partial f^{M}(n_i^M)}{\partial n_{i}^{M}} =
  -2(t_j-a_j)d^{M}(n_i^M)
  $$
  Or expressed in matrix form.
  $$
  \textbf{s}^{M} = -2d^{M}(\textbf{n}^M)
  (\textbf{t}-\textbf{a})  
  $$
</p>

<h3>Bringing it all Together</h3>

<p>
  Given a neural network with \(\textbf{M}\) layers and input \(\textbf{p}\).
  First forward propagate the input through the network.
  $$
  \textbf{a}^{0} = \textbf{p}
  $$

  $$
  \textbf{a}^{m+1} = \textbf{f}^{m+1}(\textbf{W}^{m+1}\textbf{a}^{m} +
  \textbf{b}^{m+1})
  $$

  $$
  \textbf{a} = \textbf{a}^{M}
  $$

  Where \( \textbf{a} \) is the final output of the neural network. Given this
  output let's backpropogate the sensitivities to adjust the parameters of the network based
  on the expected output \( \textbf{t} \). Compute the final layer's
  sensitivity.

  $$
  \textbf{s}^{M} = -2d^{M}(\textbf{n}^M)
  (\textbf{t}-\textbf{a})  
  $$

  Then backpropogate this sensitivity to all the other layers.

  $$
  \textbf{s}^{m} = 
  \textbf{D}^m (\textbf{n}^{m}) \left( \textbf{W}^{m + 1} \right)^{T}
  \textbf{s}^{m+1}
  $$
  For \( m = M-1 \) to \( m = 0 \). Using the sensitivities we can now update
  the parameters of the network.

  $$ \textbf{W}^{m}(k+1) = \textbf{W}^{m}(k) - \alpha \textbf{s}^{m}
  (\textbf{a}^{m-1})^{T} $$

  $$ \textbf{b}^{m}(k+1) = \textbf{b}^{m}(k) - \alpha \textbf{s}^{m} $$
</p>

<p>
  We can then train some arbitrary network to learn a function. This will
  typically take several iterations over the entire input set to obtain
  acceptable levels of accuracy. 
</p>

<p>
  Hopefully, this example of training a neural network minimizing the mean
  squared error function provided a solid basis of how neural networks work.
</p>

<p>
  To clarify, backpropagation is the step of propagating the sensitivities
  backwards in the network. SGD is the process of actually updating the
  parameters of the network.
</p>

<h2><a name="loss-functions"></a>Loss Functions</h2>

<p>
  In the example on the last page we relied on using the mean squared error for
  our <i>cost function</i>. This cost function is what we minimized through stochastic
  gradient descent (SGD) and tells how good the model is doing given the
  parameters.
</p>

<p>
  However, the mean squared error may not always be the best cost function. In
  fact a more popular loss function is the <i>cross entropy cost function.</i>
  However, before we get more into the cross-entropy cost function let's look
  into the <i>softmax classification function</i>.
</p>

<h3>Softmax Classifier</h3>

<p>
  Let's say you are building a neural network to classify between two classes.
  Our neural network will look something like the following image. Notice
  that there are two outputs \( y_1 \) and \( y_2 \) representing class one and
  two respectively. 
</p>

<img class='center-image' src='/assets/img/ml/crash_course/decision_network.png' />

<p>
  We are given a set of data points \( \textbf{X} \) and their corresponding
  labels \( \textbf{Y} \). How might we represent the labels? A given point is
  either class one or class two. The boundary is distinct. If you remember the
  linear classification boundary from earlier we said that any output greater
  than 0 was class one and any output less than 0 was class two. However, that
  does not really work here. A given data point \( \textbf{x}_i \) is simply
  class one or class two. We should not have data points be more class one than
  other data points. 
</p>

<p>
  We will use <i>one hot encoding</i> to provide labels for these points. If a
  data point has the label of class one simply assign it the label vector \(
  \textbf{y}_i = 
  \begin{bmatrix}
    1 \\
    0
  \end{bmatrix} \) and for a data point of class two assign it the label vector
  \( \textbf{y}_i = \begin{bmatrix}
    0 \\
    1
  \end{bmatrix} \)
</p>

<p>
  Say our network outputs the value \( \begin{bmatrix}
    c_1 \\
    c_2
  \end{bmatrix} \) where \(c_1, c_2 \) are just constants. We can say the
  network classified the input as class one if \( c_1 > c_2 \) or classified as
  class two if \( c_2 > c_1 \). Let's use the softmax function to interpret
  these results in a more probabilistic manner.
</p>

<p>
  The softmax function is defined as the following 
  $$
    q(\textbf{c}) = \frac{e^{c_i}}{\sum_j e^{c_j}} 
  $$

  Where \( c_i \) is the scalar output of the \(ith\) element of the output
  vector. Think of the numerator as converting the output to an un-normalized
  probability. Think of the denominator as normalizing the probability. This
  means that for every output \( i \) the loss function will have an output
  between 0 and 1. Furthermore, the sum of each output \( i \) will sum to one
  just as with any probability distribution.
</p>

<h3>Entropy</h3>

<p>
  We need to take one more step to use the softmax function as a loss function. This
  requires some knowledge of what entropy is. Think about this example. Say you
  were having a meal at EVK, one of the USC dining halls. If your meal is bad
  this event does not carry much information as the meals are almost guaranteed
  to be bad at EVK. However, if the meal is good this event carries a lot of
  information as it is out of the ordinary. You would not tell anyone about the bad meal
  because that is expected, but you would tell everyone about the good meal.
  Entropy deals with the measure of information. If we know an underlying 
  distribution \( y \) to some
  system we can define how much information is encoded in each event. We can
  write this mathematically as:
  $$
    H(y) = \sum_i y_i \log \left( \frac{1}{y_i} \right) = - \sum_i y_i \log (
    y_i )
  $$
</p>

<h3>Cross Entropy</h3>

<p>
  This definition assumes that we are operating under the correct underlying
  probability distribution. Let's say a new student at USC has no idea what the
  dining hall food is like and thinks EVK normally serves great food. This
  freshman has not been around long enough to know the true probability
  distribution of EVK food and instead assumes the probability
  distribution \( y'_i \). Now this freshman incorrectly
  thinks that bad meals are uncommon. If the freshman were to tell a 
  sophomore (who knows the real distribution) that his meal at EVK was
  bad this information would mean little to the sophomore because the
  sophomore already knows that EVK food is almost always bad. We can say that the cross
  entropy is the encoding of events in \( y \) using the wrong probability
  distribution \( y' \). This gives 
  $$
  H(y, y') = - \sum_i y_i \log y'_i
  $$
</p>

<p>
  Now let's go back to our neural network classification problem. We know the
  true probability distribution for any sample should be just the one hot
  encoded label of the sample. We also know that our generated probability
  distribution is the softmax function. This gives the final form of our cross
  entropy loss.
  $$
  L_i = -\log \left( \frac{e^{c_i}}{\sum_j e^{c_j}} \right)
  $$
  Where \( y_i = 1 \) for the correct label and \( y' \) is the softmax
  function.
  This loss function is often called the categorical cross entropy loss
  function because it works with categorical data (i.e. data that can be
  classified into distinct classes).
</p>

<p>
  And while I will not go over it here know that this function has calculable
  derivatives as well. This allows it to be used just the same as the mean
  squared error loss function in the previous example. However, the cross
  entropy loss function has many desirable properties that the mean squared
  error does not have when it comes to classification. 
</p>

<p>
  Let's say you are trying to predict the classes cat or dog. Your neural
  network has a softmax function on the output layer (as it should because this
  is a classification problem). Let's say for two inputs \(
  \textbf{x}_1,\textbf{x}_2 \) the network respectively outputs
  $$ 
  \textbf{a}_1 = 
  \begin{bmatrix}
    0.55 \\
    0.45
  \end{bmatrix},
  \textbf{a}_2 = 
  \begin{bmatrix}
    0.44 \\
    0.56
  \end{bmatrix}
  $$
  where the corresponding labels are
  $$
  \textbf{y}_1 = 
  \begin{bmatrix}
    1 \\
    0
  \end{bmatrix},
  \textbf{y}_2 = 
  \begin{bmatrix}
    0 \\
    1
  \end{bmatrix}
  $$

  As you can see the network only barely classified each result as correct. But
  by only looking at the classification error the accuracy would have been
  100%. 
</p>

<p>
  Take a similar example where the output of the network is just slightly off.
  $$ 
  \textbf{a}_1 = 
  \begin{bmatrix}
    0.51 \\
    0.49
  \end{bmatrix},
  \textbf{a}_2 = 
  \begin{bmatrix}
    0.41 \\
    0.59
  \end{bmatrix},
  \textbf{y}_1 = 
  \begin{bmatrix}
    0 \\
    1
  \end{bmatrix},
  \textbf{y}_2 = 
  \begin{bmatrix}
    1 \\
    0
  \end{bmatrix}
  $$

  Now in this case we would have a 0% classification accuracy.
</p>
  
<p>
  Let's see what our cross entropy function would have given us in each
  situation when averaged across the two samples.

  In the first situation:
  $$
  -(\log(0.55) + \log(0.56)) / 2 = 0.59
  $$

  In the second situation: 
  $$
  -(\log(0.49) + \log(0.59)) / 2 = 0.62
  $$

  Clearly this result makes a lot more sense for our situation.
</p>

<p>
  Overall, the choice of the correct loss function is dependent on the problem and is a
  decision you must make in designing your neural network. However, always keep
  in mind the general form for stochastic gradient descent will have the form:
  $$
  \mathbf{W} (k) = \textbf{W}(k-1) - \alpha \nabla J(\textbf{x}_k, \textbf{W}(k-1))
  $$
  $$
  \mathbf{b} (k) = \textbf{b}(k-1) - \alpha \nabla J(\textbf{x}_k, \textbf{b}(k-1))
  $$
  Where \( J \) is the loss function. Furthermore, the same form of
  backpropagation equations will still apply with backpropagating the
  sensitivities through the network. 
</p>


<h2><a name="regularization"></a>Regularization</h2>

<p>
  When we design a machine learning algorithm the goal is to have the algorithm
  to perform well on unseen inputs. Regularization deals with this, performing
  well on the test set which the algorithm has never seen before, sometimes at
  the cost of the training accuracy. Regularization is the process of putting
  a penalty terms in the cost function to help the model generalize to new
  inputs. Regularization does this by controlling the complexity of the model
  and preventing overfitting. 
</p>

<p>
  Given a cost function \( J(\theta, \textbf{X}, \textbf{y}) \) we can write
  the regularized version as follows. (Remember \( \theta \) notates the
  parameters of the model).
  $$
  \hat{J}(\theta, \textbf{X}, \textbf{y}) = J(\theta, \textbf{X}, \textbf{y}) +
  \alpha \Omega(\theta)
  $$

  The \( \Omega \) term is the parameter norm penalty and operates on the
  parameters of the network. The constant \( \alpha \in [0, \infty) \) controls
  the effect of the regularization on the cost function. This is a
  hyperparameter that must be tuned. Also another note, when we refer to the
  parameters of the model in regularization we typically only refer to the
  weights of the network not the biases.
</p>

<h3>\(L^2\) Parameter Regularization</h3>

<p>
  This type of regularization defines the parameter norm penalty as the
  following. 
  $$
  \Omega(\theta) = \frac{1}{2} \lVert \textbf{w} \rVert _2^2
  $$
  and the total objective function:
  $$
  \hat{J}(\theta, \textbf{X}, \textbf{y}) = J(\theta, \textbf{X}, \textbf{y}) +
  \frac{\alpha}{2} \textbf{w}^T \textbf{w}
  $$

  Evidently this regularization will penalize larger weights. In theory this
  should help prevent the model from overfitting. It is common to employ \( L^2
  \) regularization when the number of observations is less than the number of
  features. Similarly to \( L^2 \) regularization is \( L^1 \) which you can
  probably expect is just \( \Omega(\theta) = \frac{1}{2} \lVert \textbf{w}
  \rVert _1 \). In almost all cases \( L^2 \) regularization outperforms \( L^1
  \) regularization.
</p>

<h3>Early Stopping</h3>

<p>
  When we are working with a dataset we split that dataset up into testing and
  training datasets. The training dataset is to adjust the weights of the
  network. The test dataset is to check the accuracy of the model on data that
  has never been seen before.
</p>

<p>
  However, the training dataset can be divided again into the training data and
  a small subset of data called the validation set. The validation set is used
  during training to ensure the model is not overfitting. This data is not ever used
  to train the model. The validation accuracy refers to the models accuracy
  over the validation set. The goal is to minimize the validation accuracy
  through tuning hyperparameters of the network. The network is only evaluated
  on the test dataset with the fully tuned model. 
</p>

<p>
  Take a look at the below graph showing validation loss versus training loss.
  It should be clear that at a certain point the model overfits on the training
  data and begins to suffer in validation accuracy despite this not being
  reflected in the training.
</p>

<img class='center-image' src='/assets/img/ml/crash_course/valid_train_loss.png' />

<p>
  The solution to this is to simply stop training once the validation set loss
  has not improved for some time. Just like \( L^1 \) and \( L^2 \)
  regularization this is a method of decreasing overfitting on the training
  dataset. 
</p>

<h3>Ensemble Methods</h3>
<p>
  <i>Bagging</i> (short for bootstrap aggregation, a term in statistics) is the
  technique of making a model generalize better by combining multiple weaker
  learners into a stronger learner. Using this technique, several models are
  trained separately and their results are averaged for the final result. This
  ideal of one model being composed of several independent models is called an
  ensemble method. Ensemble methods are a great way to fine tune your model to
  make it generalize better on test data. Ensemble methods apply to more than
  just neural networks and can be used on any machine learning technique.
  Almost all machine learning competitions are won using ensemble methods.
  Often times these ensembles can be comprised of dozens and dozens of
  learners. 
</p>

<p>
  The idea is that if each model is trained independently of each other they
  will have their own errors on the test set. However, when the results of the
  ensemble learners are averaged the error should approach zero. 
</p>

<p>
  Using bagging we can even train a multiple models on the same dataset but be
  sure that the models were trained independently. With bagging \( k \)
  different datasets of the same size are constructed from the original dataset
  for \( k \) learners. Each dataset is constructed by sampling from the
  original dataset with some probability with replacement. So there will be
  duplicate and missing values in the constructed dataset. 
</p>

<p>
  Furthermore, differences in model initialization and hyperparameter tuning
  can make ensembles of neural networks particularly favorable. 
</p>

<h3>Dropout</h3>

<p>
  Dropout is a very useful form of regularization when used on deep neural
  networks. At a high level dropout can be thought of randomly removing neurons
  from some layer of the network with a probability \( p \). Removing certain
  neurons helps prevent the network from overfitting.
</p>

<p>
  In reality dropout is a form of ensemble learning. Dropout trains an ensemble
  of networks where various neurons have been removed and then averages the
  results, just as before. Below is an image that may help visualize what
  dropout does to a network.
</p>

<img class='center-image' src='/assets/img/ml/crash_course/dropout.jpeg' />

<p>
  Dropout can be applied to input units and hidden units. The hyperparameter of
  dropout at a given layer is the probability with which a neuron is dropped.
  Furthermore, another major benefit of dropout is that the computational cost
  of using it is relatively low. Finding the correct probability will require
  parameter tuning because a probability too low and dropout will have no
  effect, while too high and the network will be unable to learn anything. 
</p>

<p>
  Overall, dropout makes more robust models and is a standard technique
  employed in deep neural networks. 
</p>


<h2><a name="optimization-in-practice"></a>Optimization in Practice</h2>

<h3>Mini-batch Algorithm</h3>
<p>
  First let's review our SGD algorithm shown below. 
  $$
  \mathbf{\theta} (k) = \mathbf{\theta}(k-1) - \alpha \nabla J(\textbf{x}_k,
  \mathbf{\theta}(k-1))
  $$
  As this algorithm is <i>stochastic</i> gradient descent it operates one input
  example at a time. This is also referred to as online training. However, this
  is not an accurate representation of the gradient as it is only over a single
  input parameter and is not necessarily reflective of the gradient over the
  entire input space. A more accurate representation of the gradient could be
  given by the following.
  $$
  \mathbf{\theta} (k) = \mathbf{\theta}(k-1) - \alpha \nabla J(\textbf{x},
  \mathbf{\theta}(k-1))
  $$
  The gradient at each iteration is now being computed across the entire input
  space. This is referred to as batch gradient descent which we will see in a
  second is a confusing name. 
</p>

<p>
  In practice neither of these approaches are desirable. The first does not
  give good enough of an approximation of the gradient, the second is
  computationally infeasible as for each iteration the gradient of the cost
  function for the entire dataset has to be computed. <i>Mini-batch</i> methods
  are the solution to this problem.
</p>

<p>
  In mini-batch training a set of the samples are used to compute the cost
  gradient. The average of these gradients for each sample is then used. This
  approach offers a good trade off between speed and accuracy. The equation
  for this method is given below where \( Q \) is the number of samples in the
  mini-batch and \( \alpha \) is the learning rate as before.
  $$
  \mathbf{\theta} (k) = \mathbf{\theta}(k-1) - \frac{\alpha}{Q} \sum_{q=1}^{Q}\nabla
  J(\textbf{x}_q, \mathbf{\theta}(k-1))
  $$
  Remember that batch gradient descent is over the whole input space while
  mini-batch is just over a subset at a time. 
</p>

<p>
  Of course it would make sense that the samples have to be randomly drawn from
  the input space as sequential samples likely have some correlation. The
  typical procedure is to randomly shuffle the input space and the sample
  sequentially for mini-batch.
</p>

<h3>Initializations</h3>

<p>
  At this point you may be wondering how a neural network is actually
  initialized. So far the learning has been described but the actual initial
  state of the network has not been discussed. 
</p>

<p>
  You may think that how a network is initialized does not necessarily matter.
  After all the network should eventually converge to the correct parameters
  right? Unfortunately this is not the case with neural networks, the
  initialization matters greatly. Initializing to small random weights
  typically works. However, the standard for weight initialization many
  consider to be the normalized initialization method. 
</p>

<p>
  Using this method weights are randomly drawn from the following uniform
  distribution.
  $$
  \textbf{W} \sim U \left( -\frac{6}{\sqrt{m+n}}, \frac{6}{m+n} \right)
  $$
  Where \( m \) is the number of inputs into the layer and \( n \) is the 
  number of outputs from the layer. 
</p>

<p>
  As for the biases, typically just assigning them to a value of 0 works. 
</p>

<h3>Challenges in Optimization</h3>

<p>
  In classic math optimization functions we optimize some function \( f \). Now
  the same is true here where we are optimizing the loss function. However,
  keep in mind this loss function is not the same as the objective function
  which is how the model is performing on the actual inputs. This means that
  the gradient of the loss function is just an approximation of the true
  gradient of the objective function. 
</p>

<p>
  Another concern are local minima. Any deep neural network is guaranteed to
  have a very large number of local minima. Take a look at the below
  surface. This surface has two minima, a local and a maximum. If you look at
  the below contour map you can see that the algorithm converges to the local
  minimum instead of the global maximum. 
</p>

<img class='center-image' src='/assets/img/ml/crash_course/optimization_surface_minima.png' />
  
<p>
  How can we stop our neural network
  from converging to local minima? Well local minima would be a concern if the
  cost function evaluated at the local minima was far greater than the cost
  function evaluated at the global minima. It turns out that this difference is
  negligible. Most of the time, simply finding any minima is sufficient in the
  case of deep neural networks. 
</p>
  
<p>
  Another issue is saddle points, plateaus or valleys. In practice neural
  networks can escape valleys or saddle points. However, they can still pose a
  serious threat to neural networks as they can have cost functions much
  greater than the global minimum. Even more dangerous are flat regions. Small
  weights are chosen in part to avoid these flat regions in the performance
  surface. 
</p>

<p>
  In general more flat areas are problematic for the rate of convergence. It
  takes a lot of iterations for the algorithm to get over more flat regions.
  The first thought may be to increase the learning rate of the algorithm but
  too high of a learning rate will result in divergence at steeper areas of the
  performance surface. When this algorithm with a high learning rate goes
  across something like a valley it will oscillate out of control and diverge.
  An example of this is shown below. 
</p>

<img class='center-image' src='/assets/img/ml/crash_course/momentum.png' />

<p>
  At this point it should be clear that several modifications to
  backpropagation need to be made to allow solve this oscillation problem and
  to fix the learning rate issue.
</p>

<h3>Momentum</h3>

<p>
  For this concept it is useful to think of the progress of the algorithm
  as a point traveling over the performance surface. Momentum in neural
  networks is very much like momentum in physics. And since our 'particle'
  traveling the performance surface has unit mass, momentum is just the
  velocity. The equation of backprop including momentum is given by the
  following. 

  $$
  \textbf{v}(k) = \lambda \textbf{v}(k-1) - \alpha \nabla J(\textbf{x}, \mathbf{\theta}(k-1))
  $$
  $$
  \mathbf{\theta} (k) = \mathbf{\theta}(k-1) + \textbf{v}(k)
  $$
  The effect of applying this can be seen in the image below. Momentum dampens
  the oscillations and tends to make the trajectory continue in the same
  direction. Values of \( \lambda \) closer to 1 give the trajectory more momentum.
  Keep in mind \( \lambda \) itself is not momentum and is more like a force of
  friction for the particles trajectory. Typical values for \( \lambda \) are 0.5,
  0.9, 0.95 and 0.99.
</p>

<img class='center-image' src='/assets/img/ml/crash_course/momentum_working.png' />

<p>
  Nesterov momentum is an improvement on the standard momentum algorithm. With
  Nesterov momentum the gradient of the cost function is considered after the
  momentum has been applied to the network parameters at that iteration. So now
  we have:
  $$
  \textbf{v}(k) = \lambda \textbf{v}(k-1) - \alpha \nabla J(\textbf{x},
  \mathbf{\theta}(k-1) + \lambda \textbf{v}(k-1))
  $$
  $$
  \mathbf{\theta} (k) = \mathbf{\theta}(k-1) + \textbf{v}(k)
  $$
  In general, Nesterov momentum outperforms standard momentum. 
</p>

<h3>Adaptive Learning Rates</h3>

<p>
  One of the most difficult hyperparameters to adjust in neural networks is the
  learning rate. Take a look at the image below to see the effect of learning
  different learning rates on the minimization of the loss function.
</p>

<img class='center-image' src='/assets/img/ml/crash_course/learningrates.jpeg' />

<p>
  As from above we know that the trajectory of the algorithm over flat sections
  of the performance surface can be very slow. It would be nice if the
  algorithm could have a fast learning rate over these sections but a slow
  learning rate over steeper and more sensitive sections. Furthermore, the
  direction of the trajectory is more sensitive in some directions as opposed
  to others. The following algorithms will address all of these issues with
  adaptive learning rates.
</p>

<h3>AdaGrad</h3>

<p>
  The Adaptive Gradient algorithm (AdaGrad) adjusts the learning rate of each
  network parameter according to the history of the gradient with respect to
  that network parameter. This is an inverse relationship so if a given network
  parameter has had large gradients throughout the past the learning rate will
  scale down significantly. 
</p>

<p>
  Whereas before there was just one global learning rate, there is now a per
  parameter learning rate. We call the vector \( \textbf{r} \) to be the
  accumulation of the parameters past gradient squared. We initialize this term
  to zero.
  $$
  \textbf{r} = 0
  $$

  Next we compute the gradient as normal 
  $$
  \textbf{g} = \frac{1}{Q} \sum_{q=1}^{Q}\nabla J(\textbf{x}_q, \mathbf{\theta}(k-1))
  $$

  And then accumulate this gradient in \( r \) to represent the history of the
  gradient. 

  $$
  \textbf{r} = \textbf{r} + \textbf{g}^2
  $$

  And finally we compute the parameter update
  $$
  \mathbf{\theta} (k) = \mathbf{\theta}(k-1) - \frac{\alpha}{\delta +
  \sqrt{\textbf{r}}}
  \odot g
  $$

  Where \( \alpha \) is the global learning rate \( \delta \) is an extremely
  small constant ( \( 10^{-7} \) ). Notice that a element wise vector
  multiplication is being performed (by the \( \odot \) operator). Remember that each element of the gradient
  represents the partial derivative of the function with respect to a given
  parameter. The element wise multiplication will then scale the gradient with
  respect to a given parameter appropriately. The global learning rate is not a
  problem to choose and normally works as just 0.01.
</p>

<p>
  Clearly the problem with this algorithm is that it considers the whole
  sum of the squared gradient since the beginning of training. This results in
  the learning rate decreasing too much too early. 
</p>

<h3>RMSProp</h3>

<p>
  RMSProp is regarded as the goto optimization algorithm for deep neural
  networks. It is similar to AdaGrad but now there is a decay over the
  accumulation of the gradient squared so the algorithm "forgets" gradients far
  in the past. 
</p>

<p>
  As normal, compute the gradient. 
  $$
  \textbf{g} = \frac{1}{Q} \sum_{q=1}^{Q}\nabla J(\textbf{x}_q, \mathbf{\theta}(k-1))
  $$

  Now, this is where the algorithm changes with the introduction of the decay
  term \( \rho \).

  $$
  \textbf{r} = \rho \textbf{r} + (1 - \rho) \textbf{g}^2
  $$

  And the parameter update is the same.
  $$
  \mathbf{\theta} (k) = \mathbf{\theta}(k-1) - \frac{\alpha}{\delta +
  \sqrt{\textbf{r}}}
  \odot g
  $$
</p>

<h3>Second Order Algorithms</h3>

<p>
  Second order algorithms make use of the second derivative to "jump" to the
  critical points of the cost function. Further discussion of these algorithms
  is outside the scope of this tutorial. However, these algorithms do not work
  very well in practice. First of all it is computationally infeasible to
  compute the second order derivatives. Second of all, for a complex
  performance surface with many critical points it is very likely the second
  order method would go in the completely wrong direction. Overall, gradient
  descent first order methods have been shown to work better and perform
  better so I would not worry about knowing what second order algorithms are
  all about. But know that they exist and are an active area of research. 
</p>

<h2><a name="unstable-gradient"></a>The Unstable Gradient Problem</h2>

<p>
  It turns out deeper neural networks can be a lot more powerful than their
  shallow counterparts. Despite that fact that a two layer network can model
  any function it actually takes exponentially more neurons to do so than
  builder deeper layers. These deeper layers of neurons add more layers of
  abstraction for the network to work with. Deep neural networks are vital to
  visual recognition problems. Modern deep neural networks built for visual
  recognition are hundreds of layers deep. 
</p>

<p>
  However, you may take what you have learned so far and try to build a deep
  neural network. However, to your surprise you may see that adding more layers
  does not seem to help and even reduces the accuracy. Why is this the case?
</p>

<p>
  The answer is in unstable gradients. This problem plagued deep learning up
  until 2012 and is responsible for much of the deep learning boom. The cause
  of the unstable gradient problem can be formulated as different layers in the
  neural network having vastly different learning rates. And this problem only
  gets worse with the more layers that are added. The vanishing gradient
  problem is that earlier layers are learning slower than later layers. The
  exploding gradient problem is the opposite. Both of these issues deal with
  how the sensitivities are backpropagated through the network. 
</p>

<p>
  Let's recall the equation for backpropagating the sensitivities.
  $$
  \textbf{s}^{m} = 
  \textbf{D}^m (\textbf{n}^{m}) \left( \textbf{W}^{m + 1} \right)^{T}
  \textbf{s}^{m+1}
  $$
  
  Now let's say the network has 5 layers. Let's compute the various
  sensitivities recursively through the network.

  $$
  \textbf{s}^5 = \frac{\partial J}{\partial \textbf{n}^5}
  $$

  $$
  \textbf{s}^4 = d^4(\textbf{n}^4)(\textbf{W}^{5})^T \frac{\partial J}{\partial \textbf{n}^5}
  $$

  $$
  \textbf{s}^3 = d^3(\textbf{n}^3)(\textbf{W}^{4})^T d^4(\textbf{n}^4)(\textbf{W}^{5})^T \frac{\partial J}{\partial \textbf{n}^5}
  $$

  $$
  \textbf{s}^2 = d^2(\textbf{n}^2)(\textbf{W}^{3})^T d^3(\textbf{n}^3)(\textbf{W}^{4})^T d^4(\textbf{n}^4)(\textbf{W}^{5})^T \frac{\partial J}{\partial \textbf{n}^5}
  $$

  $$
  \textbf{s}^1 = d^1(\textbf{n}^1)(\textbf{W}^{2})^T d^2(\textbf{n}^2)(\textbf{W}^{3})^T d^3(\textbf{n}^3)(\textbf{W}^{4})^T d^4(\textbf{n}^4)(\textbf{W}^{5})^T \frac{\partial J}{\partial \textbf{n}^5}
  $$

  The term for \( \textbf{s}^1 \) is massive, and this is only for a five layer
  deep network. Imagine what it would be for a 100 layer deep network! The
  important take away is that all of the terms are being multiplied together. 
</p>  

<p>
  For a while the sigmoid function was believed to be a powerful activation
  function. Below is an image of the sigmoid function and its derivative. 
</p>

<img class='center-image' src='/assets/img/ml/crash_course/sigmoid.png' />

<p>
  Say we were using the sigmoid function for our five layer neural network.
  That would mean that \(\textbf{D}^1,\textbf{D}^2,\textbf{D}^3,
  \textbf{D}^4,\textbf{D}^5\) are all the derivative of the sigmoid function shown in
  red. What is the maximum value of that function? It's around 0.25. What
  types of values are we starting with for the weights? Small random values.
  The key here is that the values start small. The
  vanishing gradient problem should now start becoming clear. Because of the
  chain rule we are recursively multiplying by terms less far less than one
  causing the sensitivities to shrink and shrink going backwards in the
  network. 
</p>

<p>
  With this many so multiplication terms it would be something of a magical balancing act
  to manage all the terms so that the overall expression does not explode or
  shrink significantly.
</p>

<p>
  How do we fix this problem? The answer is actually pretty simple. Just use
  the ReLU activation function instead of the sigmoid activation function. The
  ReLU function and its derivative are shown below.
</p>

<img class='center-image' src='/assets/img/ml/crash_course/relu.png' />

<p>
  As you can see derivative is either 0 or 1 which alleviates the unstable
  gradient problem. This function is also much easier to compute. 
</p>