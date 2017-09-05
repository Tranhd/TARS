# TARS - An attempt to generate short jokes with deep learning
**Demo: [https://tarshumorbot.herokuapp.com](https://tarshumorbot.herokuapp.com)**

## Data and preprocessing 
The data is taken from a openly available short jokes dataset, downloadable from [Kaggle](https://www.kaggle.com/abhinavmoudgil95/short-jokes). The dataset contains over 200.000 short jokes ranging from 10 to 200 characters. Because of the immense size of the dataset and the fact that I'm training it on my MacBook Pro, the size to be reduced. 

The preprocessing is done by the **preprocess** class (preprocess.py). First the jokes are filtered by their length, in this model only jokes of maximum length of 12 characters is used. Then special start and end tokens are added to the jokes to know when they start and end. Next the jokes are encoded into vectors as well as shortening the vocabulary to account for 98% of the unique words as many words occur only once. 

Example output from the preprocess class:
```
Jokes loaded successfully
Jokes above a length of 12 removed
Sentence end and start tokens added and jokes word-tokenized
Found 32843 unique word tokens
Found 620930 total word tokens
Use 20424 most common words to account for the 98
 percent of the total word counts
The least frequent word in our vocabulary is 'trans-chicken' and appeared 1 time(s)
The vocabulary is of size 20426
Joke after Pre-processing: '['GO', 'how', 'to', 'get', 'a', 'cop', "'s", 'attention', 'END']'
Joke after encoding: [1, 25, 16, 51, 5, 1521, 9, 1372, 2]
```

The Tensorflow-model is fed data by the **batchgenerator** class(batchgenerator.py). The Batchgenerator divides the jokes into different buckets according to their length and pads the jokes in each bucket to the same length with a special PAD-token. This might not make such a big difference when dealing with only jokes with maximum length 12 but could speed up the computation a lot when the lengths vary since this will require to pad all jokes to the maximum length of the batch. 

## Model
The model is contained in **main.py**. The model consists of an word embedding followed by an 1-layer LSTM-network with 412 hidden nodes and finally an classifier that outputs a probability distribution over our vocabulary. The cells used were *Bidirectional* which means the network both looks into the past words but also the possible future predictions. The labels are shifted one step forward relative to the input, i.e:

input: START Time flies like an arrow, fruit flies like a banana.

label: Time flies like an arrow, fruit flies like a banana. END

When doing the prediction the distribution over possible words given the ones before (the conditional distribution of the next words given the earlier ones) are modeled according the multinomial distribution.

The model was trained using a dropout keep probability of 0.3, 0.4, 0.5 and 0.6. 0.6 have the most promising result without overfitting. 

Example output from training:
```
============================================== Epoch: 1 =============================================

100%|######################################################################|loss:   5.17 ETA:  0:00:00
Average loss:  4.2915891378
Average perplexity:  214.30682747031304

GO what is that 's only there 's ? like donut painter joke END

============================================== Epoch: 2 =============================================
100%|######################################################################|loss:   5.32 ETA:  0:00:00
Average loss:  4.01848528418
Average perplexity:  124.67560737268985

GO the talks cold they have fonzie is here END

============================================== Epoch: 3 =============================================
100%|######################################################################|loss:    4.8 ETA:  0:00:00
Average loss:  3.97033176903
Average perplexity:  115.19705854034909
Sum(pvals) > 1, argmax instead.

GO kind of him is least to the : something -anonymous END

============================================== Epoch: 4 =============================================
100%|######################################################################|loss:   4.34 ETA:  0:00:00
Average loss:  3.97282470854
Average perplexity:  112.25284564477623

GO the runaway cat jokes are the least favorite letter . END

============================================== Epoch: 5 =============================================
61%|#######################################                               |loss:   5.33 ETA:  0:30:46
```

## Purpose
The purpose of the project was to familiarize myself with recurrent neural nets and to use them within Tensorflow. Another goal was to learn about html, css and Flask to be able to deploy a demo though Heroku. The code for the Flask model is available [here](https://github.com/Tranhd/Flask_TARS)

## Conclusion 
Generating jokes is a hard and complex task since the models need to understand the semantic meaning of a joke to be able to generate new ones. Besides the difficulty of the problem my computational resources, at the moment, are severely insufficient. TARS doesn't generate jokes close to what a human is able to but can still produce reasonably correct sentences and occasionally something that resemble humor:

"What do you call a dictionary on halloween? a cheese."

"I need dead babies...... feminism."

"What is a terrorist favorite joke timing? Bloody rohypnol."

Try it out at [https://tarshumorbot.herokuapp.com](https://tarshumorbot.herokuapp.com)

## Improvement
An obvious improvement would be to use all the available data and increase the complexity of the model by a lot. Further a more sophisticated method to handle the unknown words would be desired, e.g sample from the pool of unknown words. Using another conditional distribution rather than the multinomial could also be beneficiary.

## Author

* **Ludwig Tranheden**