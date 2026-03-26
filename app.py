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
# NLTK SAFE DOWNLOAD (Render fix)
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
# EMAIL GENERATOR (RESTORED QUALITY)
# ---------------------------
def generate_email(review, sentiment, keywords):
    kw = ", ".join(keywords)

    if sentiment == "Positive":
        subject = "Thank You for Your Feedback!"
        body = f"""Dear Customer,

Thank you for your positive feedback regarding {kw}.

"{review}"

We are delighted you had a great experience and look forward to serving you again.

Best regards,
Support Team"""

    elif sentiment == "Negative":
        subject = "Apology for Your Experience"
        body = f"""Dear Customer,

We sincerely apologize for the issues related to {kw}.

"{review}"

Your feedback is important to us and we are actively working to improve.

Best regards,
Support Team"""

    elif sentiment == "Mixed":
        subject = "Thank You for Your Honest Feedback"
        body = f"""Dear Customer,

Thank you for your honest feedback.

We appreciate the positives regarding {kw}, and we acknowledge the concerns raised.

"{review}"

We are working to improve the areas you highlighted.

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
# SEND EMAIL (DEBUG SAFE)
# ---------------------------
def send_email(to_email, subject, body):
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")

    print("📤 Attempting email →", to_email)

    if not sender or not password:
        raise Exception("Email credentials missing")

    if not to_email or "@" not in to_email:
        raise Exception(f"Invalid recipient: {to_email}")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
    except Exception as e:
        print("❌ EMAIL ERROR:", e)
        raise e

# ---------------------------
# MAIN ROUTE
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
    pros = []
    cons = []
    insight_summary = ""

    if request.method == "POST":

        file = request.files.get("file")

        try:
            if file:
                df = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
                df.columns = df.columns.str.lower()
            else:
                text = request.form.get("reviews", "")
                df = pd.DataFrame({
                    "review": text.split("\n"),
                    "email": ["test@gmail.com"] * len(text.split("\n"))
                })

            if "email" not in df.columns:
                df["email"] = "test@gmail.com"

        except Exception as e:
            return f"CSV ERROR: {str(e)}"

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

            for kw in keywords:
                score = analyzer.polarity_scores(kw)['compound']
                if score > 0:
                    pros.append(kw)
                elif score < 0:
                    cons.append(kw)

    # ---------------------------
    # PROS / CONS
    # ---------------------------
    top_pros = Counter(pros).most_common(5)
    top_cons = Counter(cons).most_common(5)

    clean_pros = [p[0] for p in top_pros if "poor" not in p[0].lower()]
    clean_cons = [c[0] for c in top_cons]

    # ---------------------------
    # INSIGHTS
    # ---------------------------
    if sentiment_counts["Negative"] > sentiment_counts["Positive"]:
        insight_summary += "⚠️ Customer sentiment is mostly negative.\n"

    if clean_cons:
        insight_summary += "Main issues: " + ", ".join(clean_cons) + "\n"

    if clean_pros:
        insight_summary += "Customers appreciate: " + ", ".join(clean_pros) + "\n"

    return render_template(
        "index.html",
        results=results,
        results_json=json.dumps(results),
        sentiment_counts=sentiment_counts,
        top_pros=top_pros,
        top_cons=top_cons,
        insight_summary=insight_summary
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