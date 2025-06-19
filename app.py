__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import os, time

# Set page config
st.set_page_config(
    page_title="Semantic Book Recommender",
    page_icon="📚",
    layout="wide"
)

# Cache the data loading to avoid reloading on every interaction
@st.cache_data
def load_books_data():
    """Load and process books data"""
    # Load books data
    books = pd.read_csv("./ds/cleaned_data_3[books_with_emotions].csv")
    
    # Process thumbnail URLs
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    
    # Handle missing thumbnails with a placeholder image URL or None
    # Option 1: Use a placeholder image service
    placeholder_url = "https://via.placeholder.com/400x600/cccccc/666666?text=No+Cover"
    
    # Option 2: Use None and handle in display function
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        placeholder_url,  # or None if you want to handle it differently
        books["large_thumbnail"],
    )
    
    return books

# Cache the embedding model to avoid reloading
@st.cache_resource
def load_embedding_model():
    """Load HuggingFace embedding model that runs locally"""

    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'} )
    
    return embeddings

# Cache the vector database initialization
@st.cache_resource
def initialize_vector_db():
    """Initialize or load Chroma vector database from persistent storage"""
    
    # Define persistent directory path
    persist_directory = "./db2/"
    
    # Load embedding model
    embeddings = load_embedding_model()
    
    # Check if persistent database exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        
        status_placeholder = st.empty()
        status_placeholder.info("Loading vector database...")
        # Load existing database
        db_books = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        status_placeholder.success("Database loaded!")
        time.sleep(1)  # Show success for 1 second
        status_placeholder.empty()
    else:
        st.info("Creating new vector database...")
        # Create new database from documents
        
        # Load and split documents
        raw_documents = TextLoader("./ds/tagged_description.txt").load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        
        # Create and persist database
        db_books = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # The database is automatically persisted to disk
        st.success("Vector database created and saved!")
    
    return db_books

def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
    books_df: pd.DataFrame = None,
    db_books = None
) -> pd.DataFrame:
    """Retrieve semantic book recommendations based on query and filters"""
    
    # Perform similarity search
    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # Extract book IDs from recommendations
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    
    # Filter books based on recommendations
    book_recs = books_df[books_df["isbn13"].isin(books_list)].head(initial_top_k)
    
    # Apply category filter
    if category != "All":
        book_recs = book_recs[book_recs["simplified_category"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)
    
    # Apply tone-based sorting
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)
    
    return book_recs

def format_book_display(book_row):
    """Format book information for display"""
    description = book_row["description"]
    truncated_desc_split = description.split()
    truncated_description = " ".join(truncated_desc_split[:30]) + "..."
    
    # Format authors
    authors_split = book_row["authors"].split(";")
    if len(authors_split) == 2:
        authors_str = f"{authors_split[0]} and {authors_split[1]}"
    elif len(authors_split) > 2:
        authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    else:
        authors_str = book_row["authors"]
    
    return {
        'title': book_row['title'],
        'authors': authors_str,
        'description': truncated_description,
        'thumbnail': book_row["large_thumbnail"]
    }

def main():
    """Main Streamlit app"""
    
    # App title
    st.title("📚 Semantic Book Recommender")
    st.markdown("## Find books based on semantic similarity and emotional tone ")
    st.markdown("---")
    
    # Load data and initialize components
    with st.spinner("Loading data and models..."):
        books = load_books_data()
        db_books = initialize_vector_db()
    
    
    # Prepare dropdown options
    categories = ["All"] + sorted(books["simplified_category"].unique())
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    
    # User interface
    st.subheader("Search Parameters")
    
    # Create columns for input fields
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        user_query = st.text_input(
            "Describe the book you're looking for:",
            placeholder="e.g., A story about forgiveness, adventure in space, romantic comedy..."
        )
    
    with col2:
        category = st.selectbox("Category:", categories)
    
    with col3:
        tone = st.selectbox("Emotional Tone:", tones)
    
    # Search button
    if st.button("🔍 Find Recommendations", type="primary"):
        if user_query.strip():
            with st.spinner("Searching for recommendations..."):
                # Get recommendations
                recommendations = retrieve_semantic_recommendations(
                    query=user_query,
                    category=category,
                    tone=tone,
                    books_df=books,
                    db_books=db_books
                )
                
                # Display results
                if not recommendations.empty:
                    st.subheader(f"📖 Found {len(recommendations)} Recommendations")
                    
                    # Display books in a grid layout
                    cols_per_row = 4
                    for i in range(0, len(recommendations), cols_per_row):
                        cols = st.columns(cols_per_row)
                        
                        for j, (_, book) in enumerate(recommendations.iloc[i:i+cols_per_row].iterrows()):
                            with cols[j]:
                                book_info = format_book_display(book)
                                
                                # Display book cover with error handling
                                try:
                                    if book_info['thumbnail'] and book_info['thumbnail'] != "cover-not-found.jpg":
                                        st.image(
                                            book_info['thumbnail'], 
                                            use_container_width=True,
                                            caption=f"**{book_info['title']}**"
                                        )
                                    else:
                                        # Show placeholder for missing covers
                                        st.markdown(
                                            f"""
                                            <div style='
                                                background-color: #f0f0f0; 
                                                height: 200px; 
                                                display: flex; 
                                                align-items: center; 
                                                justify-content: center;
                                                border: 2px dashed #ccc;
                                                margin-bottom: 10px;
                                            '>
                                                <span style='color: #666;'>📚 No Cover Available</span>
                                            </div>
                                            """, 
                                            unsafe_allow_html=True
                                        )
                                        st.markdown(f"**{book_info['title']}**")
                                except Exception as e:
                                    # Fallback for any image loading errors
                                    st.markdown(
                                        f"""
                                        <div style='
                                            background-color: #f0f0f0; 
                                            height: 200px; 
                                            display: flex; 
                                            align-items: center; 
                                            justify-content: center;
                                            border: 2px dashed #ccc;
                                            margin-bottom: 10px;
                                        '>
                                            <span style='color: #666;'>📚 Cover Unavailable</span>
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                                    st.markdown(f"**{book_info['title']}**")
                                
                                # Display book details
                                st.write(f"**Authors:** {book_info['authors']}")
                                st.write(f"**Description:** {book_info['description']}")
                                st.write("---")
                else:
                    st.warning("No recommendations found. Try adjusting your search criteria.")
        else:
            st.warning("Please enter a book description to search.")
    
    # Sidebar with additional information
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses semantic search to find books similar to your description.
        
        **Features:**
        - 🔍 Semantic similarity search
        - 📁 Category filtering
        - 😊 Emotional tone sorting
        - 🤖 Runs on Local HuggingFace embeddings
        """)
        
        st.header("Tips")
        st.write("""
        - Be descriptive in your search query
        - Try different emotional tones
        - Use specific themes or plot elements
        - Experiment with different categories
        """)

if __name__ == "__main__":
    main()