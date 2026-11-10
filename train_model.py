
import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# Load data
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")
print("Fake dataset columns:", fake.columns.tolist())
print("True dataset columns:", true.columns.tolist())
print("Fake dataset size:", len(fake))
print("Real dataset size:", len(true))

# Assign labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# Combine data
data = pd.concat([fake, true], ignore_index=True)

# Prepare news text
data["text"] = (
    data["title"].fillna("").astype(str)
    + " "
    + data["text"].fillna("").astype(str)
)

data = data[["text", "label"]]
data = data[data["text"].str.strip() != ""]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# Create the NLP model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
    stop_words="english",
    max_df=0.9,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)),
("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

+
66++
6
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"]))


# Evaluate on unseen test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save trained model
os.makedirs("model", exist_ok=True)

with open("model/fake_news_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
