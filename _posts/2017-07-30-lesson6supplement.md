---
layout: post
published: true
title: "Lesson 6 Supplemental Material"
headline: ""
mathjax: true
featured: false
categories: curriculum-supplement

comments: false
---


<h3><a name="convolution-formal"></a>A Formal Definition of Convolution</h3>

<p>
  To gain a more in depth understanding of what convolution is let's take a
  look at it through a more rigorous definition. Convolution is a mathematical
  operation that has roots in signal processing. Say we are measuring the
  noisy signal \( x(t) \). Since this signal is noisy we can get a more
  accurate measurement by averaging the signal with the last several signals.
  However, more recent measurements are more relevant to the current position
  and we therefore want to have some sort of decaying weight function that
  penalizes measurements that happened a long time ago. Say \( w(a) \) is our
  weight function where \( a \) is the age of the measurement. Now say we
  wanted to get a measurement at time \( t \) taking into account this
  weighting. To do so we would have to apply the weighted average at every
  continuous moment in time before \( t \)
  $$
  s(t) = \int_{0}^{t} x(a) w(t-a) da
  $$
  This operation is the definition of convolution and is often notated as \(
  s(t) = (x*w)(t) \) The convolution function can be thought as the amount of
  overlap of functions \( x \) and \( w \). In the below image the green curve
  is the value of the convolution \( f * g \), the red is \( f \), the blue \(
  g \) and the shaded area is the product \( f(a) g(t - a) \) where \( t \) is
  the x-axis. 
</p>

<img class='center-image' src='/assets/img/ml/cnn/convgaus.gif' />

<p>
  The first argument to the convolution (in the example the function \(x(t)\))
  is the input to the function and the second (in the example the function \(
  w(t) \) ) is referred to as the kernel. 
</p>

<p>
  This continuous representation of convolution does not work for computers
  that only work with discrete values. We can convert the convolution to its
  discrete counterpart.
  $$
  s(t) = (x*w)(t) = \sum_{a=-\infty}^{\infty} x(a)w(t-a)
  $$
  However, remember that our goal is to apply this to images which have defined
  boundaries so we can constrain these infinite sums to the dimensions of the
  image.
</p>

<p>
  Furthermore, images are two dimensional so we must apply the convolution to
  a two dimensional function. 
  $$
  S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(m,n) K(i - m, j - n)
  $$
  Get used to the notation of calling \( I \) as the input image and \( K \) as
  the kernel. Furthermore, in the above equation \( n \) and \( m \) would be
  clamped to the dimensions of the image.
</p>


### CNN Architecture Examples

#### <a name="vggnet"></a>VGGNet:

The full network architecture is listed below. Note that C-64
means convolution layer with 64 filters, P-2 means pooling layer with
filter size of 2 and FC-4096 means full connected layer with 4096 neurons. 
Notice that this network follows up a block of two or
sometimes three convolution layers with a max pooling layer.

<ul>
  <li>Input: [224x224x3]</li>
  <li>C-64: [224x224x64]</li>
  <li>C-64: [224x224x64]</li>
  <li>P-2: [112x112x64]</li>
  <li>C-128: [112x112x128]</li>
  <li>C-128: [112x112x128]</li>
  <li>P-2: [56x56x128]</li>
  <li>C-256: [56x56x256]</li>
  <li>C-256: [56x56x256]</li>
  <li>C-256: [56x56x256]</li>
  <li>P-2: [28x28x256]</li>
  <li>C-512: [28x28x512]</li>
  <li>C-512: [28x28x512]</li>
  <li>C-512: [28x28x512]</li>
  <li>P-2: [14x14x512]</li>
  <li>C-512: [14x14x512]</li>
  <li>C-512: [14x14x512]</li>
  <li>C-512: [14x14x512]</li>
  <li>P-2: [7x7x512]</li>
  <li>Flatten</li>
  <li>FC-4096: [4096]</li>
  <li>FC-4096: [4096]</li>
  <li>FC-1000: [1000]</li>
  <li>Softmax for class probabilities</li>
</ul>

<img class='center-image' src='/assets/img/ml/cnn/vggnet.jpg' />