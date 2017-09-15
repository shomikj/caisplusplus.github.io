---
layout: post
published: true
title: "Environment Setup"
headline: "Setting up your environment"
mathjax: true
featured: true
categories: curriculum 

comments: false
---

<p>
  Before getting into the actual lessons we first need to set up our
  coding environment. 
</p>

<p>
  Python is the most popular tool for machine learning and it is what
  we will be using throughout this curriculum. Now you might be wondering why
  use Python for machine learning? Python is a high level programming language.
  Wouldn't things run faster if we used C++? 
</p>
  
<p>
  It turns out, we can run our machine learning programs just as fast in Python
  as we could in C++ and the simplicity of Python makes it an attractive
  choice. In a typical deep learning program all of the
  computational burden will be from large matrix multiplications. There is a
  library for Python called NumPy, a scientific computing library that offers a
  Python interface for low level, fast math operations, especially with
  matrices and other linear algebra concepts. The same is true for TensorFlow
  the deep learning package that we will be using. TensorFlow provides a Python
  interface for fast deep learning operations that can be performed on the CPU
  or GPU (the GPU is a <b>lot</b> faster at matrix multiplications than the
  CPU, making it valuable for deep learning applications). Finally, we will
  also install scikit-learn a general machine learning package for Python. 
</p>

<p>
  These instructions will walk you through what all you need to install. In
  summary we will be installing Anaconda, TensorFlow and scikit-learn all using
  Python 3.5. These instructions should work for both Windows and any Unix
  system. (However, I know Windows commonly has problems installing this
  software. I highly recommend using any Unix system if you have one, not
  including a VM).
</p>

1. A lot of the functionality in Python comes from external packages. Anaconda
   is a package manager that we will use to manage our environments and
   versions of Python. Download Anaconda from <a
   href='https://www.anaconda.com/download/'>here</a>. 
2. Check that anaconda is installed by running in your terminal <code class='language-bash'>conda info</code>
3. Create your conda environment. This will specify a version of Python and be
   a separated container for all of your Python packages to exist. Run <code
   class='language-bash'>conda create -n tfenv python=3.5</code> in your
   terminal note that this specifies to use python 3.5 and names our
   environment 'tfenv'.
4. Activate your environment. This tells your terminal session to use the
   version of Python and the packages in the conda environment. This is done
   through <code class='language-bash'>activate tfenv</code> on Windows or
   <code class='language-bash'>source activate tfenv</code> on Unix systems.
   You should see your prompt change with the name of the environment to the
   left of the input line. 
5. Make sure that pip is installed by running <code class='language-bash'>pip
   -v</code>. We will use pip to install several of the packages we will need.
   If pip is not installed follow the instructions <a
   href='https://pip.pypa.io/en/stable/installing/'>here</a>
6. Install TensorFlow. For Windows run <code class='language-bash'>pip install --ignore-installed --upgrade tensorflow</code>. For Unix based systems just run <code class='language-bash'>pip install tensorflow</code>.
7. Check that TensorFlow was actually installed. Start up the Python REPL with
   the command
   <code class='language-bash'>python</code>. When in the REPL import TensorFlow. <code class='language-python'></code>import tensorflow as tf`. At this point
   you might see some warning logs or other messages but as long as it didn't
   give an error you are good to go!
8. Install scikit-learn. Scikit-learn is on conda so we will just use
   that <code class='language-bash>conda install
   scikit-learn</code>. Once again test what we just installed. Open the Python
   REPL and try importing the package <code class='language-python'>import
   sklearn</code>.
9. NumPy should have been installed as a dependency of the other packages but
   ensure that numpy is also installed and working. Open up the Python REPL and
   type the following <code class='language-python'>import numpy as np</code>

<p>
  Finally, We will not spend time to cover the basics of Python. If you feel
  like you need to brush up on your Python go <a href='https://docs.python.org/3/tutorial/'>here</a>. We also will assume some knowledge of Calculus and Linear Algebra. If you are hesitant on these topics read through chapters 2 and 3 of <a href='https://github.com/HFTrader/DeepLearningBook/blob/master/DeepLearningBook.pdf'>this</a>.
</p>

<p>
  And that is it for now. If you have any questions please email us at
  caisplus@usc.edu. If any of these instructions do not work also email us. We
  want to update this lesson with any trouble shooting advice. 
</p>



