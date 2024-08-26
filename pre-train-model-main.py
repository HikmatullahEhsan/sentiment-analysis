import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import json
import nltk

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Step 1: Load the JSON dataset
with open('data/sample.json', 'r') as file:
    data = json.load(file)

# Convert the list of comments into a Pandas DataFrame
df = pd.DataFrame(data, columns=['comment'])

# Display the total number of comments loaded
print(f"Total number of comments: {len(df)}")

# Step 2: Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Step 3: Function to classify sentiment
def classify_sentiment(comment):
    score = sia.polarity_scores(comment)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the sentiment classifier to the dataset
df['Sentiment'] = df['comment'].apply(classify_sentiment)

# Step 4: Display the entire DataFrame
pd.set_option('display.max_rows', None)  # Show all rows
print(df)

# # Step 5: Summarize the results
# # Count the number of each sentiment
# sentiment_counts = df['Sentiment'].value_counts()

# # Calculate percentages
# total_comments = len(df)
# positive_percentage = (sentiment_counts.get('Positive', 0) / total_comments) * 100
# negative_percentage = (sentiment_counts.get('Negative', 0) / total_comments) * 100
# neutral_percentage = (sentiment_counts.get('Neutral', 0) / total_comments) * 100

# # Print the summary
# print("\nSummary of Sentiment Analysis:")
# print(f"Positive: {positive_percentage:.2f}%")
# print(f"Negative: {negative_percentage:.2f}%")
# print(f"Neutral: {neutral_percentage:.2f}%")
