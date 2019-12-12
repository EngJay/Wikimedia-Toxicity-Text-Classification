# Data

This directory contains the scripts and notebooks used to prepare the datasets for use in the experiment scripts.

Due to size constraints, the datasets themselves have been omitted but are available upon request.

## Sources

### Text Data and Word Embeddings

The source data used for this study can be downloaded from Wikimedia, Kaggle, and Google.

* **Wikipedia Personal Attacks Corpus**: The [Wikipedia Detox release](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release), commonly referred to as the Wikipedia Toxicity dataset, includes corpora of comments from the English Wikipedia talk pages, which is the forum used by Wikipedia editors to communicate with each other in regard to editing Wikipedia entries. Of the included corpora, the [Personal Attacks corpus](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release#Personal_Attacks) was the focus of this study and can be downloaded from the [figshare repo](https://figshare.com/articles/Wikipedia_Talk_Labels_Personal_Attacks/4054689).
* **Jigsaw Toxic Comments**: While ultimately not used in the final report, the Jigsaw Toxic Comments dataset from Kaggle was also explored due to its similarity to the Wikipedia Personal Attacks corpus. The data can be viewed and downloaded from the [challenge on Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
* **Google News Negative 300 Word Embeddings**: The pretrained word embeddings known as the Google News Negative 300 word embeddings were created by Google as part of the development of the word2vec algorithm. They were used for comparison with the performance of the word embeddings trained only on the Wikipedia Personal Attacks corpus and can be downloaded from the [archived word2vec project](https://code.google.com/archive/p/word2vec/). 

### Lexicons

The lexicons used in this study are listed here and the related preprocessing files can be found in the Lexicons directory within the data directory.

* **Abusive Words**:
* **AFINN-96**:
* **Bing-Liu Opinion**:
* **MSOL June15-09**:
* **NRC-EmoLex**:

## Format

TODO Explain how data would need to be formatted for use in the scripts.