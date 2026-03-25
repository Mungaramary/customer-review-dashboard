from flask import Flask, request, jsonify
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake

app = Flask(__name__)

analyzer = SentimentIntensityAnalyzer()
rake = Rake()

def analyze_review(review):
    score = analyzer.polarity_scores(review)['compound']
    
    if score >= 0.05:
        sentiment = "Positive"
    elif score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    rake.extract_keywords_from_text(review)
    keywords = rake.get_ranked_phrases()[:3]

    return sentiment, keywords

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    review = data.get("review")

    sentiment, keywords = analyze_review(review)

    return jsonify({
        "sentiment": sentiment,
        "keywords": keywords
    })

if __name__ == "__main__":
    app.run(debug=True)