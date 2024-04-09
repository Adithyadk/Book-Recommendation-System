from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset from CSV
df = pd.read_csv("Enggbooks.csv")

# Concatenate relevant columns
df['combined_text'] = df['title'].fillna('') + ' ' + df['Single_Label'].fillna('') + ' ' + df['Rest_of_Labels'].fillna('')

# Feature representation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_combined = tfidf_vectorizer.fit_transform(df['combined_text'].astype(str))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    partial_input = request.form['partial_input']
    partial_input_vector = tfidf_vectorizer.transform([partial_input])

    cosine_similarities = cosine_similarity(partial_input_vector, tfidf_matrix_combined)
    similar_books_indices = cosine_similarities.argsort()[0][-25:][::-1]

    unique_titles = set()
    recommendations = []
    for idx in similar_books_indices:
        book_id = df.loc[idx, 'idbook']
        book_title = df.loc[idx, 'title']
        if book_title not in unique_titles:
            recommendations.append({'idbook': book_id, 'title': book_title})
            unique_titles.add(book_title)

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
