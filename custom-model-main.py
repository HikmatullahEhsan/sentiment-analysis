import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import json

# Step 1: Load and prepare the labeled dataset for training
with open('data/labeled_sample.json', 'r') as file:
    train_data = json.load(file)

df_train = pd.DataFrame(train_data)

# Split the labeled dataset into training and validation sets
X_train = df_train['comment']
y_train = df_train['sentiment']

# Step 2: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Step 3: Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 4: Load the separate testing dataset (sample.json)
with open('data/sample.json', 'r') as file:
    test_data = json.load(file)

df_test = pd.DataFrame(test_data, columns=['comment'])

# Step 5: Feature extraction on the testing dataset
X_test_tfidf = vectorizer.transform(df_test['comment'])

# Predict sentiment for the testing dataset
df_test['Predicted_Sentiment'] = model.predict(X_test_tfidf)

# Step 6: Summarize the results
sentiment_counts = df_test['Predicted_Sentiment'].value_counts(normalize=True) * 100

print("\nSummary of Predicted Sentiments on Testing Data:")
print(sentiment_counts)

# Step 7: Optionally save the predictions to a new JSON file
output_data = df_test.to_dict(orient='records')
with open('data/predicted_sample.json', 'w') as file:
    json.dump(output_data, file, indent=4)

print("Predicted sentiments have been saved to 'data/predicted_sample.json'.")
