import json

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import preprocess_text


def create_vectors():

    with open("dataset.json", "r", encoding="utf-8") as file:
        news_data = json.load(file)

    documents = [
        preprocess_text(article["content"])
        for article in news_data
    ]

    vectorizer = TfidfVectorizer(stop_words='english')

    tfidf_matrix = vectorizer.fit_transform(documents)

    return vectorizer, tfidf_matrix, news_data
