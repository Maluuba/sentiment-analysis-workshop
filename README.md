# sentiment-analysis-workshop
Deep Learning for Language Workshop prepared for the AI For Social Good Summer Lab, 2018

Contains a notebook and code for developing a sentiment analysis model on a subset of [IMDB movie review data](https://www.cs.cornell.edu/people/pabo/movie%2Dreview%2Ddata/review_polarity.tar.gz) documented [here](https://www.cs.cornell.edu/people/pabo/movie%2Dreview%2Ddata/)

Pang, B., & Lee, L. (2004, July). A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd annual meeting on Association for Computational Linguistics (p. 271). Association for Computational Linguistics.

## Setup and Requirements

- Python 3.6
- TensorFlow 1.8
- Jupyter Notebook

Download and extract the pretrained model [here](https://figureqadataset.blob.core.windows.net/live-dataset/sentiment_90pct_639ep.tar.gz?sp=rl&st=2020-10-06T15:20:22Z&se=2120-10-07T15:20:00Z&sv=2019-12-12&sr=b&sig=XirNGx9sthVF48sO4Ctx64mX3prDutqOOEI8R3v7E0Y%3D).

## Sample standalone script usage:
1) train a model from scratch

`python sentiment_rnn.py train --name experiment --base-dir /my/base/dir`

2) continue training a model

`python sentiment_rnn.py train --base-dir /my/base/dir/experiment --continue-epoch 2`

3) runtime from an epoch

`python sentiment_rnn.py runtime --base-dir /my/base/dir/experiment --continue-epoch 2`
