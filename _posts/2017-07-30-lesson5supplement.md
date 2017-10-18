---
layout: post
published: true
title: "Lesson 5 Supplemental Material"
headline: ""
mathjax: true
featured: false
categories: curriculum-supplement

comments: true
---

<h2><a name="backpropagation"></a>Backpropagation Proof</h2>

In this section, we will generalize the results from the [one-layer training case](../curriculum-supplement/lesson4supplement) to
any number of layers in our network and any activation functions. This will
form the backpropagation algorithm which is used to train the parameters of a
neural network.


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
    f^{m}(n_{j}^{m})}{\partial n_{j}^{m}} 
  $$

  Writting this in vector notation we get the following:
  $$
  \frac{\partial n_{i}^{m+1}}{\partial n_{j}^{m}} = \textbf{W}^m * 
  \textbf{D}^m(\textbf{n}^m)
  $$

  Where we have \( \textbf{D}^m(\textbf{n}^m) \) be the diagonalized square
  matrix whose diagonal elements are \( \frac{\partial
    f^{m}(n_{j}^{m})}{\partial n_{j}^{m}} \)
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
  s_{i}^{M} = -2(t_j-a_j) \frac{\partial f^{M}(n_i^M)}{\partial n_{i}^{M}}
  $$
  Or expressed in matrix form.
  $$
  \textbf{s}^{M} = -2 \nabla f^{M}(\textbf{n}^M)
  (\textbf{t}-\textbf{a})  
  $$
</p>
