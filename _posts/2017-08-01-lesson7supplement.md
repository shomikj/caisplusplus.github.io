---
layout: post
published: true
title: "Lesson 7 Supplemental Material"
headline: ""
mathjax: true
featured: false
categories: curriculum-supplement

comments: false
---

### <a name="word2vec"></a>Word2Vec

A common task for RNNs is to work with word sequences. One option for doing this is to simply encode the words in our sequence using a one-hot encoding, which is a representation of words as binary vectors. Each word in the sequence would be represented as a vector whose length is equal to the number of unique words in our text. All the elements in the vector would be set to $$0$$, except the element corresponding to the word in question, which is set to $$1$$.

If are dictionary of words is of size $$n$$, then this gives $$n$$-dimensional vectors. In this $$n$$-dimensional space, each word has an equal distance from all the other words. However, this is not consistent with our intuition that some words are more closely related than others. For instance, man and woman should be “close” (meaning that the vector representations should be relatively similar) and dog and cat should be “close”. 

Instead of using one-hot encodings, what we can do is build a model to predict how close surrounding words are, and use the weights of this network as the word encodings. Visit the links below for some more in-depth descriptions:
* [https://deeplearning4j.org/word2vec](https://deeplearning4j.org/word2vec)
* [http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

### <a name="seq-to-seq"></a>Sequence-to-Sequence Learning

We have learned how to perform some regression or classification task using RNNs. We can take an input sequence and output some fixed size vector. But let’s consider a more complicated task. Say we want to learn the mapping between a sequence of variable size and another sequence of variable size. We can approach this problem by using two RNNs. 

The first RNN is the encoder and encodes the input sequence to some internal representation. The second RNN is the decoder and decodes the internal representation to an output sequence.

A clear usage of these networks are in translation tasks. They are actually what power Google Translate. They are also often used for chatbots. To learn more about sequence to sequence models visit the following link: [https://www.tensorflow.org/tutorials/seq2seq](https://www.tensorflow.org/tutorials/seq2seq). 
