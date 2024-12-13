## Pre requisites:

Python

pip - command line tool for python

## To install necessary libraries:
  
pip install scikit-learn nltk

Note: Please disconnect from any VPN connections before running the program. Executing the program for first time might take lot of time. But subsequent execution will be quick. I have disabled SSL in the program to avoid any hindrances

## Text Classification with Naive Bayes: Program Explanation

### Introduction
This program demonstrates how to perform **text classification** using the **Naive Bayes algorithm** with the **20 Newsgroups dataset**. The goal is to classify news articles into one of 20 predefined categories (e.g., `alt.atheism`, `rec.sport.baseball`). This process involves preprocessing the text data, training a classifier, and evaluating its performance.

---

### Program Breakdown

#### **1. Importing Required Libraries**

```python
import nltk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
```
- `nltk`: A library for natural language processing (e.g., downloading stopwords).
- `datasets`: Provides access to the 20 Newsgroups dataset.
- `train_test_split`: Splits the data into training and testing sets.
- `CountVectorizer`: Converts text data into numerical features using the bag-of-words model.
- `MultinomialNB`: Implements the Naive Bayes classification algorithm.
- `accuracy_score`, `classification_report`: Evaluate the model's performance.

---

#### **2. Downloading NLTK Data**

```python
nltk.download('stopwords')
```
- Downloads the `stopwords` dataset from NLTK, which contains common words like "the" and "and" that are usually removed in preprocessing.

---

#### **3. Loading the 20 Newsgroups Dataset**

```python
newsgroups = datasets.fetch_20newsgroups(subset='all')
```
- The **20 Newsgroups dataset** contains 20 categories of news articles.
- `subset='all'` loads the entire dataset (training + testing).

---

#### **4. Splitting Data into Training and Testing Sets**

```python
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.3, random_state=42)
```
- `X_train` and `y_train`: Training data and labels.
- `X_test` and `y_test`: Testing data and labels.
- `test_size=0.3`: Reserves 30% of the data for testing.
- `random_state=42`: Ensures reproducibility of results.

---

#### **5. Preprocessing: Bag-of-Words Model**

```python
vectorizer = CountVectorizer(stop_words='english')
X_train_dtm = vectorizer.fit_transform(X_train)  # Learn vocabulary and transform training data
X_test_dtm = vectorizer.transform(X_test)       # Transform test data using the same vocabulary
```
- `CountVectorizer`: Converts text into numerical features using the **bag-of-words** model.
  - `stop_words='english'`: Removes common English words to focus on meaningful terms.
- `fit_transform`: Learns the vocabulary from training data and transforms it into a document-term matrix (DTM).
- `transform`: Transforms the test data using the same vocabulary learned from the training data.

---

#### **6. Training the Naive Bayes Classifier**

```python
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_dtm, y_train)
```
- `MultinomialNB`: A Naive Bayes classifier suitable for discrete features like word counts.
- `fit`: Trains the model on the document-term matrix (`X_train_dtm`) and corresponding labels (`y_train`).

---

#### **7. Making Predictions**

```python
y_pred = nb_classifier.predict(X_test_dtm)
```
- `predict`: Uses the trained model to predict the categories of the test data (`X_test_dtm`).

---

#### **8. Evaluating the Model**

```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```
- `accuracy_score`: Calculates the percentage of correct predictions out of all predictions.

```python
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))
```
- `classification_report`: Provides detailed metrics for each category:
  - **Precision**: Proportion of correctly predicted instances out of all instances predicted for a class.
  - **Recall**: Proportion of correctly predicted instances out of all actual instances of a class.
  - **F1-Score**: Harmonic mean of precision and recall.
  - **Support**: Number of actual instances for each category in the test set.

---

### Example Output

After running the program, you will see:

#### **Accuracy**:
```
Accuracy: 78.10%
```

#### **Classification Report**:
```
                        precision    recall  f1-score   support

             alt.atheism       0.89      0.83      0.86       319
           comp.graphics       0.78      0.83      0.80       389
 comp.os.ms-windows.misc       0.74      0.67      0.70       394
    rec.autos                  0.88      0.88      0.88       397
...
```

---

### Categories in the Dataset

The 20 Newsgroups dataset contains the following categories:

1. `alt.atheism` - Discussions on atheism.
2. `comp.graphics` - Topics about computer graphics.
3. `comp.os.ms-windows.misc` - Miscellaneous topics about Windows OS.
4. `comp.sys.ibm.pc.hardware` - IBM PC hardware discussions.
5. `comp.sys.mac.hardware` - Macintosh hardware topics.
6. `comp.windows.x` - Topics about the X Window System.
7. `misc.forsale` - Items for sale.
8. `rec.autos` - Automobile discussions.
9. `rec.motorcycles` - Motorcycle discussions.
10. `rec.sport.baseball` - Topics about baseball.
11. `rec.sport.hockey` - Topics about hockey.
12. `sci.crypt` - Cryptography discussions.
13. `sci.electronics` - Electronics-related topics.
14. `sci.med` - Medical discussions.
15. `sci.space` - Space and astronomy topics.
16. `soc.religion.christian` - Christianity and religious topics.
17. `talk.politics.guns` - Topics about politics and guns.
18. `talk.politics.mideast` - Middle East politics discussions.
19. `talk.politics.misc` - Miscellaneous political topics.
20. `talk.religion.misc` - Miscellaneous religious discussions.

---

### Summary

This program demonstrates the entire workflow of a text classification task:
- **Loading Data**: Using the 20 Newsgroups dataset.
- **Preprocessing**: Converting text to numerical features with a bag-of-words model.
- **Model Training**: Using a Naive Bayes classifier.
- **Evaluation**: Measuring accuracy and detailed classification metrics.

This foundational knowledge can be extended to more advanced techniques (e.g., word embeddings, deep learning) for better performance in complex scenarios.
