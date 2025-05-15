import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# this loads data from a cvs file
df = pd.read_csv("IMDB Dataset.csv")

# this map and converts the sentiment column to binary values
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Defining negation words
negation_words = ['not', "don't", "didn't", "isn't", "wasn't", "aren't", "weren't", "won't", "wouldn't", 
                  "shouldn't", "can't", "couldn't", "no", "never", "none", "nor", "n't"]

# this is to apply the preprocessing function to the review column
def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s']", "", text)  # remove non-alphabetic characters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
    return text

df['cleaned'] = df['review'].apply(preprocess)

# function to handle custom negation
# this function will handle negation by appending "not_" to the next word
def handle_negation(text):
    words = text.split()
    result = []
    negate = False
    for word in words:
        if word in negation_words:
            negate = True
            result.append(word)
        elif negate:
            result.append("not_" + word)
            negate = False
        else:
            result.append(word)
    return " ".join(result)

# Data without negation handling
df['without_negation'] = df['cleaned']

# Data with negation handling
df['with_negation'] = df['cleaned'].apply(handle_negation)

# this helps to split the data into training and testing sets
X_train_wo, X_test_wo, y_train, y_test = train_test_split(df['without_negation'], df['sentiment'], test_size=0.2, random_state=42)
X_train_w, X_test_w, _, _ = train_test_split(df['with_negation'], df['sentiment'], test_size=0.2, random_state=42)

# this is to create a TF-IDF vectorizer to convert the text data into numerical features
vectorizer_wo = TfidfVectorizer(max_features=5000)
vectorizer_w = TfidfVectorizer(max_features=5000)

X_train_vec_wo = vectorizer_wo.fit_transform(X_train_wo)
X_test_vec_wo = vectorizer_wo.transform(X_test_wo)

X_train_vec_w = vectorizer_w.fit_transform(X_train_w)
X_test_vec_w = vectorizer_w.transform(X_test_w)

# this Train model without negation
model_wo = LogisticRegression(max_iter=200)
model_wo.fit(X_train_vec_wo, y_train)
predictions_wo = model_wo.predict(X_test_vec_wo)

print("\n=== Without Negation Handling ===")
print(f"Accuracy: {accuracy_score(y_test, predictions_wo):.4f}")
print(classification_report(y_test, predictions_wo))

# this Train model with negation
model_w = LogisticRegression(max_iter=200)
model_w.fit(X_train_vec_w, y_train)
predictions_w = model_w.predict(X_test_vec_w)

print("\n=== With Negation Handling ===")
print(f"Accuracy: {accuracy_score(y_test, predictions_w):.4f}")
print(classification_report(y_test, predictions_w))

# these are the Custom test samples
custom_samples = [
    "I did not like this movie at all.",
    "This is not good.",
    "I don't think it's worth it.",
    "I absolutely loved the storyline!",
    "The movie was amazing and emotional."
]

# Preprocess and apply negation
samples_cleaned = [preprocess(text) for text in custom_samples]
samples_negated = [handle_negation(text) for text in samples_cleaned]

samples_vec_wo = vectorizer_wo.transform(samples_cleaned)
samples_vec_w = vectorizer_w.transform(samples_negated)

print("\n--- Custom Test Cases ---")
for i, text in enumerate(custom_samples):
    pred_wo = model_wo.predict(samples_vec_wo[i])[0]
    pred_w = model_w.predict(samples_vec_w[i])[0]
    sentiment_wo = "positive" if pred_wo == 1 else "negative"
    sentiment_w = "positive" if pred_w == 1 else "negative"
    print(f"\nInput: {text}")
    print(f"Without Negation Handling: {sentiment_wo}")
    print(f"With Negation Handling   : {sentiment_w}")
