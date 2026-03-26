from flask import Flask, render_template, request, send_file, jsonify
import nltk
import os

# ---------------------------
# NLTK FIX
# ---------------------------
nltk_data_path = "/opt/render/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
import pandas as pd
import io
from collections import Counter
import json
import csv
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)

analyzer = SentimentIntensityAnalyzer()
rake = Rake()

# ---------------------------
# SENTIMENT
# ---------------------------
def analyze_review(review):
    scores = analyzer.polarity_scores(review)
    compound = scores['compound']
    pos = scores['pos']
    neg = scores['neg']

    if pos > 0.15 and neg > 0.15:
        sentiment = "Mixed"
    elif compound >= 0.2:
        sentiment = "Positive"
    elif compound <= -0.2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    rake.extract_keywords_from_text(review)
    keywords = rake.get_ranked_phrases()[:3]

    return sentiment, keywords

# ---------------------------
# EMAIL GENERATION
# ---------------------------
def generate_email(review, sentiment, keywords):
    kw = ", ".join(keywords)

    if sentiment == "Positive":
        subject = "Thank You!"
        body = f"Thanks for your feedback on {kw}.\n\n{review}"
    elif sentiment == "Negative":
        subject = "Apology"
        body = f"Sorry about {kw}.\n\n{review}"
    elif sentiment == "Mixed":
        subject = "We Appreciate Your Feedback"
        body = f"We appreciate your feedback on {kw}.\n\n{review}"
    else:
        subject = "Feedback Received"
        body = f"Thanks for your feedback.\n\n{review}"

    return subject, body

# ---------------------------
# SEND EMAIL (SAFE)
# ---------------------------
def send_email(to_email, subject, body):
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")

    if not sender or not password:
        raise Exception("Email credentials missing")

    if not to_email or "@" not in to_email:
        raise Exception(f"Invalid email: {to_email}")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.send_message(msg)

# ---------------------------
# MAIN
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
    pros, cons = [], []

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            df = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
            df.columns = df.columns.str.lower()
        else:
            text = request.form.get("reviews", "")
            df = pd.DataFrame({
                "review": text.split("\n"),
                "email": ["test@gmail.com"] * len(text.split("\n"))
            })

        for _, row in df.iterrows():
            review = str(row.get("review", ""))
            email = str(row.get("email", ""))

            if not review.strip():
                continue

            sentiment, keywords = analyze_review(review)
            subject, body = generate_email(review, sentiment, keywords)

            results.append({
                "review": review,
                "sentiment": sentiment,
                "keywords": ", ".join(keywords),
                "subject": subject,
                "email": body,
                "to": email
            })

            sentiment_counts[sentiment] += 1

    return render_template(
        "index.html",
        results=results,
        results_json=json.dumps(results),
        sentiment_counts=sentiment_counts,
        top_pros=[],
        top_cons=[]
    )

# ---------------------------
# SEND SINGLE EMAIL
# ---------------------------
@app.route("/send_email", methods=["POST"])
def send_email_route():
    try:
        data = request.get_json()
        send_email(data["to"], data["subject"], data["body"])
        return jsonify({"status": "sent"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ---------------------------
# SEND ALL EMAILS
# ---------------------------
@app.route("/send_all", methods=["POST"])
def send_all():
    try:
        data = request.get_json()
        for r in data:
            send_email(r["to"], r["subject"], r["email"])
        return jsonify({"status": "sent"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ---------------------------
# DOWNLOAD CSV
# ---------------------------
@app.route("/download")
def download():
    data = json.loads(request.args.get("data"))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Review", "Sentiment", "Keywords", "Email"])

    for r in data:
        writer.writerow([r["review"], r["sentiment"], r["keywords"], r["email"]])

    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="report.csv"
    )

if __name__ == "__main__":
    app.run(debug=True)