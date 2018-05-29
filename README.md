# sentiment-analysis-workshop
Deep Learning for Language Workshop prepared for the AI For Social Good Summer Lab, 2018

Contains a notebook and code for developing a sentiment analysis model on a subset of [IMDB movie review data](https://www.cs.cornell.edu/people/pabo/movie%2Dreview%2Ddata/review_polarity.tar.gz) documented [here](https://www.cs.cornell.edu/people/pabo/movie%2Dreview%2Ddata/)

Pang, B., & Lee, L. (2004, July). A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd annual meeting on Association for Computational Linguistics (p. 271). Association for Computational Linguistics.

## Setup and Requirements

- Python 3.6
- TensorFlow 1.8
- Jupyter Notebook

Download and extract the pretrained model from [https://msrmtl-public-store.azureedge.net/ai4good/sentiment_90pct_639ep.tar.gz](https://msrmtl-public-store.azureedge.net/ai4good/sentiment_90pct_639ep.tar.gz).

## Sample standalone script usage:
1) train a model from scratch

`python sentiment_rnn.py train --name experiment --base-dir /my/base/dir`

2) continue training a model

`python sentiment_rnn.py train --base-dir /my/base/dir/experiment --continue-epoch 2`

3) runtime from an epoch

`python sentiment_rnn.py runtime --base-dir /my/base/dir/experiment --continue-epoch 2`
