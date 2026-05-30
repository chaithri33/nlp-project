import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
with open("dataset.json", "r", encoding="utf-8") as file:
    news_data = json.load(file)

# Extract contents
documents = [article["content"] for article in news_data]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(documents)

print("News Reader Analysis Chatbot")
print("Type 'exit' to stop")

while True:

    query = input("\nEnter your question: ")

    if query.lower() == "exit":
        break

    # Convert query to TF-IDF
    query_vector = vectorizer.transform([query])

    # Cosine Similarity
    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )

    best_match = similarity_scores.argmax()

    print("\nRelevant News:")
    print(news_data[best_match]["title"])
    print(news_data[best_match]["content"])
