import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake

# Load data
df = pd.read_csv("reviews.csv")

# ---------------------------
# 1. ADVANCED SENTIMENT (VADER)
# ---------------------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["review"].apply(get_sentiment)

# ---------------------------
# 2. EXTRACT KEY PHRASES (RAKE)
# ---------------------------
rake = Rake()

def extract_keywords(text):
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:3]  # top 3 phrases

df["keywords"] = df["review"].apply(extract_keywords)

# ---------------------------
# 3. SMART PROS & CONS
# ---------------------------
pros = []
cons = []

for _, row in df.iterrows():
    if row["sentiment"] == "Positive":
        pros.extend(row["keywords"])
    elif row["sentiment"] == "Negative":
        cons.extend(row["keywords"])

from collections import Counter

top_pros = Counter(pros).most_common(5)
top_cons = Counter(cons).most_common(5)

# ---------------------------
# 4. SMART EMAIL GENERATOR
# ---------------------------
def generate_email(review, sentiment, keywords):
    if sentiment == "Positive":
        return f"""
        Subject: Thank You for Your Feedback!

        Dear Customer,

        We're thrilled to hear your feedback about {", ".join(keywords)}.

        "{review}"

        Thank you for choosing us. We look forward to serving you again!

        Best regards,
        Support Team
        """

    elif sentiment == "Negative":
        return f"""
        Subject: Apology for Your Experience

        Dear Customer,

        We sincerely apologize for the issues related to {", ".join(keywords)}.

        "{review}"

        We are actively working to improve this area.

        Please reach out so we can make this right.

        Best regards,
        Support Team
        """

    else:
        return f"""
        Subject: Thank You for Your Feedback

        Dear Customer,

        Thank you for your feedback regarding {", ".join(keywords)}.

        "{review}"

        We appreciate your input.

        Best regards,
        Support Team
        """

df["email_response"] = df.apply(
    lambda x: generate_email(x["review"], x["sentiment"], x["keywords"]), axis=1
)

# Save output
df.to_csv("output_results.csv", index=False)

print("Top Pros:", top_pros)
print("Top Cons:", top_cons)