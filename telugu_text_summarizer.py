
import streamlit as st
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.lang.te import Telugu
from indicnlp.tokenize import sentence_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
import networkx as nx

nlp_model = Telugu()

def extract_text_from_url(url):
    """
    Scrapes the webpage content, extracting the title and paragraph text.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1')
        title = title_tag.get_text().strip() if title_tag else 'No title available'
        paragraphs = soup.find_all('p')
        content = ' '.join(p.get_text().strip() for p in paragraphs)
        return {
            'title': title,
            'content': content
        }
    else:
        st.error(f"Failed to fetch the webpage. Status code: {response.status_code}")
        return None

def generate_summary(text):
    """
    Summarizes the given text using a PageRank-based algorithm.
    """
    doc = nlp_model(text)
    stopwords = [] 

    word_freq = {}
    sentences = sentence_tokenize.sentence_split(text, lang='te')


    for sentence in sentences:
        for word in sentence.split():
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1

    max_freq = max(word_freq.values(), default=1)

    for word in word_freq:
        word_freq[word] /= max_freq

    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(sentences)
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(bow_matrix)
    tfidf_csr = csr_matrix(tfidf_matrix)

    graph = nx.Graph()
    for i, sentence in enumerate(sentences):
        graph.add_node(i, sentence=sentence)

    for i in range(tfidf_csr.shape[0]):
        for j in range(tfidf_csr.shape[1]):
            weight = tfidf_csr[i, j]
            if weight > 0:
                graph.add_edge(i, j, weight=weight)

    sentence_scores = nx.pagerank(graph)

    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    text_length = len(text)
    if text_length < 500:
        num_sentences = 1
    elif 500 <= text_length < 1500:
        num_sentences = 3
    elif 1500 <= text_length < 3000:
        num_sentences = 4
    else:
        num_sentences = 5

    summary = ' '.join([sentences[i] for i in ranked_sentences[:num_sentences]])
    return summary

st.title("Telugu Text Summarizer")

input_url = st.text_input("Enter the URL of the page to scrape and summarize:")

if input_url:
    data = extract_text_from_url(input_url)
    if data:
        st.subheader("Title")
        st.write(data['title'])

        st.subheader("Full Content")
        st.write(data['content'])

        summarized_text = generate_summary(data['content'])

        st.subheader("Summary")
        st.write(summarized_text)
