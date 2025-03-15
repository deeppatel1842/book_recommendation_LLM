import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
#from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import gradio as gr

load_dotenv('../env/.env')

books = pd.read_csv('../dataset/books_With_emotions.csv')

books['large_thumbnail'] = books['thumbnail'] + '&fife=w800'
books['small_thumbnail'] = books['thumbnail'] + '&fife=w200'
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    "../images/cover-not-found.jpg",
    books['large_thumbnail'],
)

raw_documents = TextLoader('../dataset/tagged_description.txt',encoding='utf-8').load()
text_Splitter = CharacterTextSplitter(chunk_size=0,chunk_overlap=0,separator ='\n')
documents = text_Splitter.split_documents(raw_documents)
# Load the Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(
    documents,
    embedding= embedding_model
)

def retrive_semantic_recommendations(
        query:str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books['isbn13'].isin(books_list[:final_top_k])]

    if category != 'All':
        book_recs = book_recs[book_recs['categories'] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by='joy',ascending=False,inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by='surprise',ascending=False,inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by='anger',ascending=False,inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by='fear',ascending=False,inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by='sadness',ascending=False,inplace=True)

    return book_recs

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrive_semantic_recommendations(query,category,tone)
    result = []

    for _,row in recommendations.iterrows():
        description = row['description']
        truncated_desc_split = description.split()
        truncated_description = ' '.join(truncated_desc_split[:30]) + '...'

        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else: 
            authors_str = row['authors']

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        result.append((row['large_thumbnail'], caption))

    return result

#categories = ['All'] + sorted(books['categories'].unique())
categories = ['All'] + [
    'Fiction',
    'Juvenile Fiction',
    'Biography & Autobiography',
    'History',
    'Literary Criticism',
    'Philosophy',
    'Religion',
    'Comics & Graphic Novels',
    'Drama',
    'Juvenile Nonfiction',
    'Science',
    'Poetry'
]
tones = ['All'] + ['Happy','Surprising','Angry','Suspenseful','Sad']

#dashboard

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a book description of a book:",
                                placeholder='e.g., A story about forgiveness')
    
        category_dropdown = gr.Dropdown(choices=categories, label= "Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label='Select a tone:', value="All")

        submit_button = gr.Button("Find recommendation")

    gr. Markdown('## Recommendations')
    output = gr.Gallery(label='Recommended books', columns=8, rows=2)

    submit_button.click(
        fn = recommend_books, 
        inputs=[user_query,category_dropdown,tone_dropdown],
        outputs= output)
    
if __name__ == '__main__':
    dashboard.launch()
