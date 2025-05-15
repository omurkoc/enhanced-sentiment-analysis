import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# this loads data from a cvs file 
df = pd.read_csv("IMDB Dataset.csv")

# this map and converts the sentiment column to binary values  
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Preprocessing function here we define a function to clean the text data
# and remove unwanted characters, HTML tags, and URLs
def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s']", "", text)  # remove non-alphabetic characters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
    return text

# this is to apply the preprocessing function to the review column
df['cleaned'] = df['review'].apply(preprocess)

# thid helps to split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['sentiment'], test_size=0.2, random_state=42)

# this is to create a TF-IDF vectorizer to convert the text data into numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Training a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Predictions
predictions = model.predict(X_test_vec)

# Evaluation metrics
print("\n=== Normal Sentiment Analysis (No Negation Handling) ===")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions))

# Test custom sentences
custom_samples = [
    "I did not like this movie at all.",
    "This is not good.",
    "I don't think it's worth it.",
    "I absolutely loved the storyline!",
    "The movie was amazing and emotional."
]

custom_cleaned = [preprocess(text) for text in custom_samples]
custom_vec = vectorizer.transform(custom_cleaned)

print("\n--- Custom Test Cases ---")
for i, text in enumerate(custom_samples):
    pred = model.predict(custom_vec[i])[0]
    sentiment = "positive" if pred == 1 else "negative"
    print(f"Input: {text}")
    print(f"Predicted Sentiment: {sentiment}")
