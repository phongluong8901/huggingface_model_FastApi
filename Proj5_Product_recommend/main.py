from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

#load a pre-trained sentecce-Bert model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Load the ebbedings and product data
df = pd.read_pickle('./models/product_embeddings.pkl')

df['imgs'] = df['imgs'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Function to get recommendations
def recommend_products(query, top_k=10):
    query = query.lower()
    query_embedding = model.encode(query)
    df['similarity'] = df['embeddings'].apply(lambda x: cosine_similarity([query_embedding], [x]).flatten()[0])
    recommendations = df.sort_values(by='similarity', ascending=False).head(top_k)
    return recommendations[['title', 'brand', 'category', 'similarity', 'imgs']]


#app
app = Flask(__name__)


#routes
@app.route('/', methods=['GET','POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        query = request.form['query']
        recommendations = recommend_products(query).to_dict(orient='records')
    return render_template('index.html', recommendations=recommendations)
        

#python main
if __name__ == "__main__":
    app.run(debug=True)