import nltk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK datasets
nltk.download('stopwords')

# Load the 20 Newsgroups dataset
newsgroups = datasets.fetch_20newsgroups(subset='all')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.3, random_state=42)

# Convert the text data into a matrix of token counts (bag-of-words model)
vectorizer = CountVectorizer(stop_words='english')  # Remove common words like 'the', 'and', etc.
X_train_dtm = vectorizer.fit_transform(X_train)  # Learn vocabulary and transform the training data
X_test_dtm = vectorizer.transform(X_test)  # Transform the test data

# Create a Naive Bayes classifier and train it
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_dtm, y_train)

# Predict the categories for the test data
y_pred = nb_classifier.predict(X_test_dtm)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display classification report (precision, recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

