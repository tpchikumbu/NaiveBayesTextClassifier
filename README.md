### Introduction
This project implements a Naive Bayes classifier for sentiment analysis using the AfriSenti dataset. The dataset consists of tweets labeled as positive, negative, or neutral, covering 14 African languages. For this assignment, the Hausa dataset was chosen due to its size and balance.

### Installations, Imports, and Downloads
The necessary Python modules are installed and imported, including `jax`, `numpy`, `pandas`, `matplotlib`, and `scikit-learn`. The AfriSenti dataset is downloaded from the GitHub repository if not already present in the current directory.

### Data Loading & Preprocessing
The dataset is loaded into pandas dataframes for training, validation, and testing. Neutral examples are discarded to focus on binary classification (positive or negative). The text is cleaned by replacing URLs with a `[URL]` token, numbers with a `[NUM]` token, and removing extra whitespaces.

### Vocabulary Construction
Two tokenization methods are explored: whitespace tokenization and Byte Pair Encoding (BPE). The vocabulary is constructed by mapping each unique word or subword to an index. The vocabulary is represented using three variables:
- `index2type`: List of unique types in the vocabulary.
- `type2index`: Dictionary mapping types to their index.
- `type2count`: Dictionary mapping types to their token occurrences in the training data.

### Text Vectorization
The text data is vectorized using one-hot encoding, where each sentence is represented as a binary vector. The output labels are converted to numerical representations (positive = 1, negative = 0).

### Naive Bayes Classifier
A Naive Bayes classifier is implemented to predict the sentiment of the tweets. The frequency of each token in the two classes is calculated and used to compute the conditional probabilities. The classifier is trained on the training data and evaluated on the development data.

### Evaluation
The model's performance is evaluated using Scikit-learn's `classification_report` function. The results are compared with Scikit-learn's Naive Bayes classifier as a benchmark.

### Results
The classifier's performance is reported for different values of the smoothing parameter `alpha`. The results include precision, recall, and F1-score for both tokenization methods (whitespace and BPE).

### Conclusion
This project demonstrates the implementation of a Naive Bayes classifier for sentiment analysis on the AfriSenti dataset. The use of different tokenization methods and vectorization techniques is explored, and the model's performance is evaluated using standard metrics.

### References
- AfriSenti GitHub Repository: [https://github.com/afrisenti-semeval/afrisent-semeval-2023](https://github.com/afrisenti-semeval/afrisent-semeval-2023)