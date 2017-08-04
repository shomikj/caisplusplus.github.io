---
layout: post
published: true
title: "Lesson 4 Supplemental Material"
headline: ""
mathjax: true
featured: false
categories: curriculum-supplement

comments: false
---


<h2>Training a Neural Network</h2>

<p>
  Let's consider the simple neural network below that has two inputs to a
  single neuron.
</p>

<img class='center-image' src='/assets/img/ml/crash_course/adaline_two_input.png' />

<p>
  Now let's use our network to classify generic inputs 
  \(   \begin{bmatrix}
    \textbf{p}_1 \\
    \textbf{p}_2 
  \end{bmatrix} \). We can begin to construct this classifier by saying that
  for any output \( a > 0 \) the input can be categorized as class one and for any output \( a \le
  0\) the input can be categorized as class two. We can find the decision
  boundary by setting the equation of the network equal to zero. 
  $$ \textbf{wp} + b = 0 $$
  $$ w_1 p_1 + w_2 p_2 + b = 0$$
  $$ w_2 p_2 = - w_1 p_1 - b $$
  $$ p_2 = - \frac{w_1}{w_2} p_1 - \frac{b}{w_2} $$

  It should be clear that this is in the form \( y = mx + b \). Let's graph
  this equation.
</p>

<img class='center-image' src='/assets/img/ml/crash_course/adaline_decision.png' />

<p>
  All of the area covered in gray is for class one and all the white area is
  class two. This is a basic linear classifier. Think of it like linear
  regression but instead of continuous outputs its output is a category or
  class.
</p>

<p>
  We will now discover an algorithm that can be used to find the line to best
  separate data points. Take the example below where a line is discovered to
  best separate the blue and the red data points. We want to algorithmically
  find this line. 
</p>

<img class='center-image' src='/assets/img/ml/crash_course/linear_class.png' />

<p>
  Let's simplify how we represent this equation by putting all of the
  parameters that are to be adjusted in one term. Note that \( p_1 \) and \(
  p_2 \) cannot be adjusted but are given. Say that \( \textbf{z} =
  \begin{bmatrix}
    \textbf{w} \\
    b
  \end{bmatrix} \) and \( \textbf{x} = 
  \begin{bmatrix}
    \textbf{p} \\
    1
  \end{bmatrix} \) notice that the dot product of these two vectors gives the
  same result as before 

  $$ a = \textbf{x}^T \textbf{z} = \textbf{wp} + b $$

  Transposing the \( \textbf{x} \) term is to simply make the product of \(
  \textbf{x} \) and \( \textbf{z} \) possible to compute and while keeping both
  vectors as column vectors. Remember the rules of matrix multiplication!
</p>

<p>
  We can say that the error is the value we <i>expected</i> subtracted by the
  <i>actual</i> output of the network. Say \( t \) was the value expected and \( a \) was the
  output of the network. The error would then be \( e = t - a \). A common measure
  for error is the mean square error. This is given by the expected value of
  the squared error. The expected value here is the expected value of a given
  input given consideration of all input/output pairs. Mathematically we can
  write this as

  $$ J(\textbf{x}) = E[e^2] = E[(t-a)^2] = E[(t-\textbf{x}^T \textbf{z})^2] $$

  Expanding this expression then gives the following. Note that we can use the
  linearity of expected value to split the expected value terms apart.
  $$ J(\textbf{x}) = E[t^2 - 2 t \textbf{x}^T \textbf{z} + \textbf{x}^T \textbf{z} \textbf{z}^T \textbf{x} ] $$

  Note that in the last term all of the funky business done with the transposes is just to make
  the matrix multiplication work out. \( \textbf{x}^T \textbf{z}
  \textbf{x}^T \textbf{z} \) is not a valid product but \( \textbf{x}^T \textbf{z}
  \textbf{z}^T \textbf{x} \) is. Next apply use the linearity of expected
  value.

  $$ J(\textbf{x}) = E[t^2] - 2\textbf{x}^T E[t\textbf{z}] + \textbf{x}^T
  E[\textbf{z}\textbf{z}^T]\textbf{x} $$

  Let's just simplify the form a little bit by making the following
  substitutions.
  $$ J(\textbf{x}) = c - 2\textbf{x}^T\textbf{h} + \textbf{x}^T\textbf{Rx} $$
  where
  $$ c = E[t^2], \textbf{h} = E[t\textbf{z}], \textbf{R} = E[\textbf{z} \textbf{z}^T] $$
</p>

<p>
  It turns out this is in the form of a quadratic function. Finding the minimum
  point of this quadratic function will minimize the mean square error which is
  our goal. We can do this by finding the gradient of the function and
  setting it equal to \( 0 \).

  $$ \nabla J(\textbf{x}) = \nabla ( c - 2\textbf{x}^T\textbf{h} +
  \textbf{x}^T\textbf{Rx} ) = -2\textbf{h} + 2\textbf{Rx} = 0$$

  $$ \textbf{R} \textbf{x} = \textbf{h} $$
  $$ \textbf{R}^{-1} \textbf{R} \textbf{x} = \textbf{R}^{-1} \textbf{h} $$

  $$ \textbf{x} = \textbf{R}^{-1} \textbf{h} $$
</p>

<p>
  So \( \textbf{R}^{-1} \textbf{h} \) is the minimum point of our mean square
  error \( J(\textbf{x} ) \). However, in practice we do not actually compute
  these values \(\textbf{R}^{-1}\) and \(\textbf{h}\) but instead approximate
  the gradient of \( J(\textbf{x}) \) numerically. We can do so through the
  Least Mean Squares LMS algorithm. We will also shortly see that
  approximating this numerically will have a number of other benefits when
  expanding to more complex functions.
</p>

<p>
  The gradient of \( J(\textbf{x}) \) is given by:

  $$ J(\textbf{x}) = (t(k) - a(k))^2 = e^{2} (k) $$
  $$ \nabla J(\textbf{x}) = \nabla e^{2} (k) $$

  Where \( k \) is the iteration number. Instead of computing the expected
  value of the squared error we are approximating it at some iteration \( k \) 
  This approximation of the gradient is called stochastic gradient. And
  we descend the gradient looking for the minimum, giving rise to the important term in machine learning
  called <b>Stochastic Gradient Descent (SGD)</b>. Our derivatives are with respect to
  the network parameters, the bias and the weights. 
</p>

<p>
  As there are many weights for our neuron, the following expression gives the
  gradient of our term evaluated at the arbitrary weight \( j \). Say there are
  \( R \) inputs (and therefore weights) to this neuron we are considering.
  
  $$ [ \nabla e^2(k) ]_j = \frac{\partial e^2 (k)}{\partial w_j} $$

  We can then apply the chain rule.

  $$ \frac{\partial e^2 (k)}{\partial w_j} = 2 e(k) \frac{\partial e(k)}{\partial w_j} $$

  Do the same for the bias term, the other parameter of our network.

  $$ \frac{\partial e^2 (k)}{\partial b} = \frac{\partial e^2 (k)}{\partial b}
  = 2 e(k) \frac{\partial e(k)}{\partial b} $$
</p>  

<p>
  Now evaluate the term \( \frac{\partial e(k)}{\partial w_j} \) 

  $$ \frac{\partial e(k)}{\partial w_j} = \frac{\partial (t(k) - 
  a(k))}{\partial w_j}  = \frac{\partial}{\partial w_j} ( t(k) - (\textbf{w}^T
  \textbf{p}(k) + b))$$

  \(p_i(k)\) is \( i \)th element of the input at the \( k \)th iteration. 

  $$ \frac{\partial}{\partial w_j} \left( t(k) - \left( \sum_{i=1}^{R} w_i
  p_i(k) +b \right) \right) $$

  Many of the terms in this expression are not dependent on the network weights
  \( w_j \). \( \frac{\partial t(k) }{\partial w_j} = 0 \) and \(
  \frac{\partial b}{\partial w_j} = 0 \). Furthermore, \( \frac{\partial w_i
  p_i}{\partial w_j} = 0\) for any \(i \ne j\). The only nonzero term of this
  expression comes out to be \( \frac{\partial (w_j p_j)}{\partial w_j} = p_j \)
  but the negative sign in front of the summation makes \( -p_j \). We now have

  $$ \frac{\partial e(k)}{\partial w_j} = -p_j(k) $$

  When taking the derivative of the expression with respect to the bias \( b
  \), we can see the only term dependent on \( b \) is \( b \) itself. This
  just gives \( 1 \). 

  $$ \frac{\partial e(k)}{\partial b} = -1 $$
</p>

<p>
  To simplify writing the final formula let's go back to our previous notation
  of using \( \textbf{z} \) to represent the parameters of the network. 
  $$ 
  \textbf{z}(k) = 
  \begin{bmatrix}
    p_j(k) \\
    1
  \end{bmatrix} 
  $$

  We can now rewrite the original equation for the gradient of the squared
  error. 

  $$ \nabla J(\textbf{x}) = \nabla e^2 (k) = -2e(k)\textbf{z}(k) $$

  At iteration \( k \) all we have to do is multiply the input \( p_j \) by the
  error.
</p>

<p>
  Our next step is to "descend" this gradient to find the minimum points. We
  know that our gradient is in direction \( \nabla J(\textbf{x}) \) therefore,
  for a given point \( \textbf{x}_k \) the direction of the minimum point is \(
  \textbf{x}_k - \alpha \nabla J( \textbf{x}_k) \) where \( \alpha \) is how
  far we go down the gradient. So we can say that the next iteration should be
  the previous iteration after taking this step in the direction of the
  gradient. We can write this as follows.
</p>

<p>
  $$ \textbf{x}_{k+1} = \textbf{x}_k + 2 \alpha e(k) \textbf{z}(k) $$

  We say that \( \alpha \) is the learning rate. 
  This can then be split up into the weight and bias update terms.

  $$ \textbf{w}(k+1) = \textbf{w}(k) + 2\alpha e(k) \textbf{p}(k) $$
  $$ b(k+1) = b(k) + 2\alpha e(k) $$
</p>

<p>
  Let's now write that equation for the general case when there could be many
  neurons in the layer such as with the image below. Note that the activation
  function \( f \) in this case is the linear activation function.
</p>

<img class='center-image' src='/assets/img/ml/crash_course/neuron_layer.png' />

<p>
  Simply look at a given "row" \( i \).

  $$ \textbf{w}_{i}(k+1) = \textbf{w}_{i}(k) + 2\alpha e_{i}(k) \textbf{p}(k) $$
  $$ b_{i}(k+1) = b_{i}(k) + 2\alpha e_{i}(k) $$

  Where \( e_{i} \) is the error of the \( i \)th row of the error in the
  layer's output. And then we can just write it in matrix form to clean things
  up. 

  $$ \textbf{W}(k+1) = \textbf{W}(k) + 2\alpha \textbf{e}(k) \textbf{p}(k) $$
  $$ \textbf{b}(k+1) = \textbf{b}(k) + 2\alpha \textbf{e}(k) $$
</p>

<p>
  And that's all we need to train our network. Using this algorithm the network
  would be able to converge to optimal network parameters. However, we are limited to
  single layer networks and linear activation functions. In the next section we
  will examine how to generalize this to any number of layers and any
  activation function. 
</p>