from sklearn.metrics.pairwise import cosine_similarity

from vectorizer import create_vectors

from preprocess import preprocess_text


vectorizer, tfidf_matrix, news_data = create_vectors()


def get_response(user_query):

    query = preprocess_text(user_query)

    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )

    best_match = similarity_scores.argmax()

    return news_data[best_match]
