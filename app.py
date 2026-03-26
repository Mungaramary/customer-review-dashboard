from flask import Flask, render_template, request, jsonify, send_file
import nltk
import os
import io
import json
import csv
import pandas as pd
from collections import Counter
import smtplib
from email.mime.text import MIMEText
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake

app = Flask(__name__)

# ---------------------------
# NLTK FIX (FOR RENDER)
# ---------------------------
nltk_data_path = "/opt/render/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

for pkg in ["stopwords", "punkt"]:
    try:
        nltk.data.find(pkg)
    except:
        nltk.download(pkg, download_dir=nltk_data_path)

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
# EMAIL GENERATOR
# ---------------------------
def generate_email(review, sentiment, keywords):
    kw = ", ".join(keywords)

    if sentiment == "Positive":
        subject = "Thank You for Your Feedback!"
        body = f"""Dear Customer,

Thank you for your positive feedback regarding {kw}.

"{review}"

We are delighted you had a great experience.

Best regards,
Support Team"""

    elif sentiment == "Negative":
        subject = "Apology for Your Experience"
        body = f"""Dear Customer,

We sincerely apologize for the issues related to {kw}.

"{review}"

We are actively working to improve.

Best regards,
Support Team"""

    elif sentiment == "Mixed":
        subject = "Thank You for Your Honest Feedback"
        body = f"""Dear Customer,

Thank you for your honest feedback.

We appreciate the positives regarding {kw} and acknowledge your concerns.

"{review}"

We are improving the highlighted areas.

Best regards,
Support Team"""

    else:
        subject = "Thank You for Your Feedback"
        body = f"""Dear Customer,

Thank you for your feedback.

"{review}"

Best regards,
Support Team"""

    return subject, body

# ---------------------------
# SEND EMAIL
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
# MAIN ROUTE (SAFE DEBUG)
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    try:
        results = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
        pros = []
        cons = []
        insight_summary = ""

        if request.method == "POST":

            file = request.files.get("file")

            if not file:
                return "❌ Please upload a CSV file"

            # READ CSV SAFELY
            df = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
            print("📊 RAW COLUMNS:", df.columns)

            df.columns = df.columns.str.strip().str.lower()
            print("✅ CLEAN COLUMNS:", df.columns)

            if "review" not in df.columns:
                return "❌ CSV must contain a 'review' column"

            if "email" not in df.columns:
                return "❌ CSV must contain an 'email' column"

            for _, row in df.iterrows():
                review = str(row["review"])
                email = str(row["email"])

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

                for kw in keywords:
                    score = analyzer.polarity_scores(kw)['compound']
                    if score > 0:
                        pros.append(kw)
                    elif score < 0:
                        cons.append(kw)

        # ---------------------------
        # CHART DATA
        # ---------------------------
        top_pros = Counter(pros).most_common(5)
        top_cons = Counter(cons).most_common(5)

        # ---------------------------
        # INSIGHTS
        # ---------------------------
        if sentiment_counts["Negative"] > sentiment_counts["Positive"]:
            insight_summary += "⚠️ Customer sentiment is mostly negative.\n"

        if top_cons:
            insight_summary += "Main issues: " + ", ".join([c[0] for c in top_cons]) + "\n"

        if top_pros:
            insight_summary += "Customers appreciate: " + ", ".join([p[0] for p in top_pros]) + "\n"

        return render_template(
            "index.html",
            results=results,
            results_json=json.dumps(results),
            sentiment_counts=sentiment_counts,
            top_pros=top_pros,
            top_cons=top_cons,
            insight_summary=insight_summary
        )

    except Exception as e:
        print("🔥 ERROR:", e)
        return f"🔥 Internal Error: {str(e)}"

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