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

Follow these steps.

1. Download Anaconda from <a href='https://www.anaconda.com/download/'>here</a>
2. Check that anaconda is installed by running `conda info`
3. Create your conda environment. `conda create -n tfenv python=3.5` note that
   this specifies to use python 3.5 and names our enviornment 'tfenv'.
4. Activate your environment. This is done through `activate tfenv` on Windows
   or `source activate tfenv` on Unix systems. You should see your prompt
   change with the name of the environment to the left of the input line. 
5. Make sure that pip is installed by running `pip -v`. If pip is not installed
   follow the instructions <a href='https://pip.pypa.io/en/stable/installing/'>here</a>
6. Install TensorFlow. For Windows run `pip install --ignore-installed --upgrade tensorflow`. For Unix based systems just run `pip install tensorflow`.
7. Check that TensorFlow was actually installed. Start up the Python REPL
   `python`. Then import TensorFlow. `import tensorflow as tf`. At this point
   you might see some warning logs or other messages but as long as it didn't
   give an error you are good to go!
