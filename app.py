from flask import Flask, render_template, request, send_file, jsonify
import nltk
import os

nltk_data_path = "/opt/render/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download ALL required resources
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
# SENTIMENT FUNCTION
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
    keyword_text = ", ".join(keywords)

    if sentiment == "Positive":
        subject = "Thank You for Your Feedback!"
        body = f"""Dear Customer,

Thank you for your positive feedback regarding {keyword_text}.

"{review}"

We’re glad you had a great experience.

Best regards,
Support Team"""
    elif sentiment == "Negative":
        subject = "Apology for Your Experience"
        body = f"""Dear Customer,

We sincerely apologize for the issues related to {keyword_text}.

"{review}"

We are working to improve this.

Best regards,
Support Team"""
    elif sentiment == "Mixed":
        subject = "Thank You for Your Honest Feedback"
        body = f"""Dear Customer,

We appreciate your feedback regarding {keyword_text}.

"{review}"

We are improving the areas you highlighted.

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
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS")

    if not sender_email or not sender_password:
        raise ValueError("Email environment variables not set")

    if not to_email:
        raise ValueError("Recipient email missing")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

# ---------------------------
# MAIN ROUTE
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

            # ---------------------------
            # CSV HANDLING
            # ---------------------------
            if file and file.filename.endswith(".csv"):
                try:
                    # Handle encoding issues
                    try:
                        content = file.stream.read().decode("UTF8")
                    except:
                        content = file.stream.read().decode("latin1")

                    stream = io.StringIO(content)
                    df = pd.read_csv(stream)

                    print("📊 CSV Columns:", df.columns)

                    # Normalize columns
                    df.columns = df.columns.str.strip().str.lower()

                    # Auto-fix common column names
                    df.rename(columns={
                        "review text": "review",
                        "reviews": "review",
                        "email address": "email"
                    }, inplace=True)

                    if "review" not in df.columns:
                        return f"❌ CSV must contain a 'review' column. Found: {list(df.columns)}"

                    if "email" not in df.columns:
                        df["email"] = "nimumungara@gmail.com"

                    reviews = df.to_dict(orient="records")

                except Exception as e:
                    print("❌ CSV ERROR:", e)
                    return f"CSV Error: {str(e)}"

            else:
                # TEXT INPUT
                reviews_text = request.form.get("reviews") or ""
                reviews = [{"review": r, "email": "nimumungara@gmail.com"} for r in reviews_text.split("\n")]

            # ---------------------------
            # PROCESS DATA
            # ---------------------------
            for row in reviews:
                review = str(row.get("review", ""))
                email_to = str(row.get("email", ""))

                if not review.strip():
                    continue

                sentiment, keywords = analyze_review(review)
                subject, email_body = generate_email(review, sentiment, keywords)

                results.append({
                    "review": review,
                    "sentiment": sentiment,
                    "keywords": ", ".join(keywords),
                    "subject": subject,
                    "email": email_body,
                    "to": email_to
                })

                sentiment_counts[sentiment] += 1

                for kw in keywords:
                    score = analyzer.polarity_scores(kw)['compound']
                    if score > 0:
                        pros.append(kw)
                    elif score < 0:
                        cons.append(kw)

        # ---------------------------
        # INSIGHTS
        # ---------------------------
        top_pros = Counter(pros).most_common(5)
        top_cons = Counter(cons).most_common(5)

        clean_pros = [p[0] for p in top_pros if "poor" not in p[0].lower()]
        clean_cons = [c[0] for c in top_cons]

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

    except Exception as e:
        print("🔥 FULL ERROR:", e)
        return f"🔥 Internal Error: {str(e)}"

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

# ---------------------------
# SEND EMAIL ROUTE
# ---------------------------
@app.route("/send_email", methods=["POST"])
def send_email_route():
    data = request.json

    try:
        send_email(data["to"], data["subject"], data["body"])
        return jsonify({"status": "sent"})
    except Exception as e:
        print("❌ EMAIL ERROR:", e)
        return jsonify({"status": "error", "message": str(e)})

# ---------------------------
# RUN APP
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)