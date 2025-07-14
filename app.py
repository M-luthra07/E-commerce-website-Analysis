from flask import Flask, render_template, request
from textblob import TextBlob
import pandas as pd
from wordcloud import WordCloud
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import io
import base64
import numpy as np
app = Flask(__name__, template_folder="template", static_folder="static")
DATA_PATH = "online_review.csv"
df = pd.read_csv(DATA_PATH)
df = df[['Review', 'Rating']].dropna().head(200)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

def predict_rating(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.6:
        return 5
    elif polarity > 0.3:
        return 4
    elif polarity > 0:
        return 3
    elif polarity > -0.3:
        return 2
    else:
        return 1

def true_sentiment(rating):
    return "Positive" if rating >= 4 else "Negative" if rating <= 2 else "Neutral"

def generate_wordcloud():
    static_folder = os.path.join(app.root_path, "static")
    wordcloud_path = os.path.join(static_folder, 'wordcloud.png')

    if os.path.exists(wordcloud_path):
        return

    all_text = ' '.join(df['Review'].dropna().astype(str).map(clean_text))
    wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    os.makedirs(static_folder, exist_ok=True)
    wc.to_file(wordcloud_path)

def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close()
    return img_base64

def plot_confusion_matrix():
    y_true = []
    y_pred = []
    for _, row in df.iterrows():
        y_true.append(true_sentiment(row['Rating']))
        y_pred.append(get_sentiment(clean_text(str(row['Review']))))
    labels = ["Positive", "Neutral", "Negative"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    return plot_to_base64()

def plot_bar_chart():
    y_true = []
    y_pred = []
    for _, row in df.iterrows():
        y_true.append(true_sentiment(row['Rating']))
        y_pred.append(get_sentiment(clean_text(str(row['Review']))))

    labels = ["Positive", "Neutral", "Negative"]
    actual_counts = [y_true.count(label) for label in labels]
    predicted_counts = [y_pred.count(label) for label in labels]

    x = np.arange(len(labels))  # label locations
    width = 0.35

    plt.figure(figsize=(7,4))
    plt.bar(x - width/2, actual_counts, width, label='Actual', color='orange')
    plt.bar(x + width/2, predicted_counts, width, label='Predicted', color='green')
    plt.xticks(x, labels)
    plt.ylabel('Count')
    plt.title('Actual vs Predicted Sentiment Counts')
    plt.legend()

    return plot_to_base64()

def plot_roc_curve():
    # For ROC, consider only Positive and Negative classes (ignore Neutral)
    y_true = []
    y_scores = []

    for _, row in df.iterrows():
        true_label = row['Rating']
        sentiment = get_sentiment(clean_text(str(row['Review'])))

        # Binary: Positive=1, Negative=0; skip Neutral
        if true_sentiment(true_label) == "Neutral":
            continue

        y_true.append(1 if true_sentiment(true_label) == "Positive" else 0)

        # Use polarity as score
        blob = TextBlob(clean_text(str(row['Review'])))
        polarity = blob.sentiment.polarity
        y_scores.append(polarity)

    if len(y_true) < 2:
        # Not enough data for ROC
        return None

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    return plot_to_base64()

@app.route('/')
def index():
    generate_wordcloud()
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.form['review']
    cleaned_review = clean_text(review)
    sentiment = get_sentiment(cleaned_review)
    rating = predict_rating(cleaned_review)

    correct = sum(
        get_sentiment(clean_text(str(row['Review']))) == true_sentiment(row['Rating'])
        for _, row in df.iterrows()
    )
    accuracy = round((correct / len(df)) * 100, 2)

    cm_img = plot_confusion_matrix()
    bar_img = plot_bar_chart()
    roc_img = plot_roc_curve()

    return render_template('result.html',
                           review=review,
                           sentiment=sentiment,
                           rating=rating,
                           accuracy=accuracy,
                           cm_img=cm_img,
                           bar_img=bar_img,
                           roc_img=roc_img)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 5000)





