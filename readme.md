# 📚 Semantic Book Recommender
### An intelligent book recommendation system that uses semantic search and emotional tone analysis to help users discover their next great read. Built with Streamlit, HuggingFace embeddings, and ChromaDB for fast, accurate recommendations.
####  Features

- Semantic Search: Find books based on natural language descriptions
- Emotional Tone Filtering: Sort recommendations by emotional mood (Happy, Sad, Suspenseful, etc.)
- Category Filtering: Browse by book categories
- Local AI Models: Uses HuggingFace embeddings that run locally (no API keys needed)
- Persistent Vector Database: Fast loading with ChromaDB persistence
- Responsive UI: Built with Streamlit


## 🚀 Live Demo

[**Try the app on Streamlit Cloud**](https://todo)

## ⚙️ How It Works

1. **Describe** what kind of book you're looking for
2. **Filter** by category and emotional tone (optional)
3. **Discover** personalized recommendations with book covers and descriptions

### Example Queries
- "A story about forgiveness and redemption"
- "Space adventure with alien encounters"
- "Romantic comedy set in a small town"
- "Mystery thriller with unexpected twists"

## 📁 Project Structure

```
├── app.py
├── db
│   └── chroma.sqlite3 #based on google gen ai embeddings
├── db2
│   └── chroma.sqlite3 #based on huggingface embeddings
├── ds
│   ├── cleaned_data_3[books_with_emotions].csv #final dataset
│   └── tagged_description.txt #descriptions with tags to generate embeddings
├── eda_and_cleaning.ipynb # EDA and preprocessing
├── readme.md
├── requirements.txt # pip freeze 
├── sentiment-analysis.ipynb # Applies a transformer-based emotion classifier to book desc.
├── text-classification.ipynb # assign simplified categories to book desc. with Huggingface pipeline
└── vector-search.ipynb
```
> [refer here for requirements](#️-requirements-to-inspect-the-project)
## 🔟 Data Workflow

1. **EDA & Cleaning**
   - Loads raw data and inspects for missing values and outliers.
   - Drops records with missing critical fields (description, num_pages, rating, year).
   - Removes descriptions with fewer than 20 words.
   - Handles missing subtitles by concatenating with titles.
   - Prepares a tagged description column for vector database use.

2. **Category Simplification & Classification**
   - Maps original categories to a simplified set.
   - Uses zero-shot classification to predict missing categories.
   - Evaluates classifier accuracy on known samples.
   - Updates the dataset with predicted categories.

3. **Emotion Analysis**
   - Splits descriptions into sentences and applies an emotion classifier.
   - Aggregates maximum emotion scores per book.
   - Merges emotion scores into the main dataset.

---

## 🗒️ Requirements To Inspect the Project:

- langchain-google-genai (if using /db/ which is google embedding)
- langchain-community
- streamlit
- pandas
- numpy
- langchain-community
- langchain-huggingface
- langchain-text-splitters
- langchain-chroma
- sentence-transformers
- transformers
- torch (comes with transformers)
- matplotlib (prep. notebooks only)
- plotly (prep. notebooks only)