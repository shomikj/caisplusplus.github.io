---
layout: post
published: true
title: "Backprop By Hand"
headline: "Keeping it Old School"
mathjax: true
featured: false
categories: curriculum-supplement

comments: true
---

After going through our workshop on neural networks and getting
some feedback from our members, it became evident that some additional
theory review would be valuable to allow the concepts behind neural
networks to really sink in. 

In order to make this review as thorough as possible,
we decided to take some inspiration from Matt Mazur's [blog post](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) on backpropagation
and to walk through one complete iteration of the neural network training process
by hand: complete with concrete numbers, calculations and all.

**Note**: Before going through this post, it may be helpful to have first
gone through our original lessons on the theory behind neural networks. 
Links to those lessons can be found here: [Neural Networks Part 1 (Architecture)](/blog/curriculum/lesson4), 
[Part 2 (Training)](/blog/curriculum/lesson5). It may also be helpful
to have the Part 2 (Training) lesson open side-by-side when going through this post,
in case you want to quickly reference any of the equations we discussed there.

### Table of Contents:
1. [Problem Setup](#problem)
2. [Preprocessing](#preprocess)
3. [Forward Propagation](#forward)
4. [Backpropagation](#backprop)
5. [Gradient Descent](#grad)
6. [Forward Propagation, Redux](#forward2)

## <a name="problem"></a>Problem Setup

Suppose we want to train a small neural network to predict whether a student
will pass an upcoming test. We want this neural network to take in
$$\text{hours slept}$$ and $$\text{hours studied}$$ as inputs, and to output a value
between $$0$$ and $$1$$ that represents the probability that the student will pass.

We are given only one training example, and some initial random values 
for our neural network weights. (The neural network has one hidden layer
and out output layer, so there will be two weight matrices in total.
For simplicity's sake, we'll go ahead and ignore biases in this example.)

We are also given some information on the average and standard deviation
of our inputs (hours slept and hours studied), which we will use to
standardize our inputs before feeding them into the network. And lastly,
we're told that we should use the *sigmoid* activation function, which we'll denote 
with $$f(x)$$, on both our hidden layer and output layer, and *cross entropy loss* ($$J$$) to gauge how well/poorly our network is doing. Our problem setup, then, is as follows:

<img src='/images/backprop-by-hand/setup1.jpg' style="width: 65%;"/>

**Disclaimer:** we wrote the *categorial cross-entropy* loss function
on the blackboard, when technically we should have wrote the *binary
cross-entropy* loss function. Binary cross entropy is used
when we are trying to predict only a 0 or 1 
(i.e. predicting between 2 classes, which is what we want), and categorical
cross entropy (a  generalization of binary cross-entropy) is used when trying to predict between multiple classes via a one-hot vector
(as was the case with MNIST/handwritten digits). They both do essentially the same thing,
but the binary cross-entropy loss function equation is formulated a little bit differently,
as you'll see in the example at the end of the first forward propagation step.

Next, a couple notation refreshers:
* $$\textbf{z}^m$$: a vector representing the weighted sum of the $$m$$th layer's neurons. ($$L$$ denotes the last layer.)
* $$\textbf{a}^m$$: activation of the $$mth$$th layer's neurons. We calculate this by passing each element in $$\textbf{z}^m$$ through the activation function.
* $$\delta^m$$: error/sensitivity of the $$m$$th layer's neurons. This tells us how the final cost ($$J$$) will change if we shift any of the weighted sums in $$\textbf{z}^m$$ a tiny bit in the positive direction.
* $$\frac{\partial J}{\partial \textbf{W}^{m}}$$: gradient of the $$m$$th layer's weights. This tells us how the final cost will change if we shift any of the $$m$$th layer's weights a tiny bit in the positive direction.

And lastly, the equations that we'll be using for forward propagation,
backpropagation, and gradient descent:
<br/>
<img src='/images/backprop-by-hand/setup2.jpg' style="width: 65%;"/>

You can check out a derivation of the backpropagation/gradient descent equations in our [Neural Network Training](/blog/curriculum/lesson5) lesson. We've moved some of the terms around within the equations
to get the dimensions to match up for our matrix operations, but the numbers themselves
will be the same in either case.

## <a name="preprocess"></a>Step 0: Preprocessing
First, we'll use the given mean/standard deviation data
to standardize our inputs. This makes sure that our neural network
can compare apples to apples, instead of apples to lemons/honeydew/etc.

<img src='/images/backprop-by-hand/preprocess.jpg' style="width: 50%;"/>

## <a name="forward"></a>Step 1: Forward Propagation
Next, we'll feed these standardized inputs forward through each layer of the neural network, applying a series of matrix multiplications and activation functions along the way,
to generate our initial prediction. We'll then calculate the (binary cross-entropy) loss using this prediction to see if it improves after going through a full training step.

<img src='/images/backprop-by-hand/forwardprop.jpg' style="width: 60%;"/>

## <a name="backprop"></a>Step 2: Backpropagation
Since you can see that we made the wrong initial prediction (*failed*, instead of
*passed*), we'll now go ahead and find out how we can tweak our neural network
so that it can do a better job of making predictions. To do this, we'll work our way
backward through the network and calculate the *error* ($$\delta$$), or *sensitivity*, of each
layer of neurons. Each of these error terms represents the partial derivative
of the final cost ($$J$$) with respect to the current neuron's weighted sum ($$z$$).

<img src='/images/backprop-by-hand/backprop.jpg' style="width: 60%;"/>

## <a name="grad"></a>Step 3: Gradient Descent
After calculating all the sensitivities, we can use these sensitivities to find
the actual gradient of the parameters (i.e. weights) in our network.

<img src='/images/backprop-by-hand/grad-descent.jpg' style="width: 60%;"/>


Then, we'll use our calculated gradients to apply gradient descent. This entails
subtracting the gradient (multiplied by our current learning rate) from the original weights, so that we move down the *cost surface* in a way that should *decrease* our final cost.

<img src='/images/backprop-by-hand/grad-descent-update.jpg' style="width: 60%;"/>

## <a name="forward2"></a>Step 4: Forward Propagation, Redux
After going through one training iteration, we should test to make sure that
our neural network's predictions are more accurate than before. We can do this
by using our new weights to once again feed the original input through the network,
generating a new prediction, and then re-calculating the loss based on this prediction.

<img src='/images/backprop-by-hand/forward2.jpg' style="width: 50%;"/>

Since our prediction is now closer to the actual $$y$$ value ($$1$$), and since the loss is lower than before, we can conclude that the training step was successful! If we had more data, we'd repeat these steps a bunch of times over the entire training set (likely taking *average* cost and gradient values over *batches* of the data) to fully train our neural network. 

For now though, hopefully this concrete example helped clarify any lingering confusions you may have had with the neural network training process.