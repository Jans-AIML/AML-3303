"""
Campus FAQ Chatbot - Streamlit Application
This standalone app creates an interactive chatbot UI for Lambton College FAQs
"""

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sys

# Set page configuration
st.set_page_config(
    page_title="Campus FAQ Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import FAQ data
sys.path.append('.')
from week4_rag_data import faq_data, faq_texts

# Initialize session state for caching models
@st.cache_resource
def load_model_and_index():
    """Load the sentence transformer model and build FAISS index"""
    # Create lines list from faq_texts
    lines = [qa.strip() for qa in faq_texts if qa.strip()]
    
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings
    embeddings = model.encode(lines)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    return model, index, lines

# Load resources
model, index, lines = load_model_and_index()

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2em;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .question-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .confidence-high {
        color: green;
        font-weight: bold;
    }
    .confidence-low {
        color: orange;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎓 Campus FAQ Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions about Lambton College</div>', unsafe_allow_html=True)

# Sidebar for information
with st.sidebar:
    st.header("📚 About This Chatbot")
    st.info("""
    This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer your questions about Lambton College.
    
    It searches through our FAQ database to find the most relevant answers using semantic similarity.
    """)
    
    st.header("💡 Tips")
    st.markdown("""
    - Ask natural questions like "What's the tuition cost?"
    - Try different phrasings for the same question
    - If the answer isn't helpful, try rephrasing your question
    """)
    
    st.header("📊 FAQ Categories")
    categories = [
        "🎓 Admissions",
        "📚 Programs", 
        "🏫 Campus Facilities",
        "💰 Tuition & Financial Aid",
        "🤝 Student Services",
        "📅 Academic Calendar",
        "🎯 Career Services",
        "🎉 Student Life",
        "📞 Contact & Locations"
    ]
    for cat in categories:
        st.markdown(f"- {cat}")

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    user_question = st.text_input(
        "Ask your question:",
        placeholder="e.g., What are the admission requirements?",
        help="Type your question about Lambton College"
    )

with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

# Process and display results
if user_question and search_button:
    st.markdown("---")
    
    # Create embedding for user question
    q_emb = model.encode([user_question])
    
    # Search for top 3 results
    distances, indices = index.search(np.array(q_emb), k=3)
    
    # Display question
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.markdown(f"**Your Question:** {user_question}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display best answer
    st.markdown("### 🎯 Best Match Answer:")
    best_idx = indices[0][0]
    best_distance = distances[0][0]
    
    # Calculate a simple confidence score (lower distance = higher relevance)
    # Typical distances range from 0-2 for semantic search, normalize to percentage
    confidence = max(0, 100 - (best_distance * 50))
    
    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    st.markdown(lines[best_idx])
    
    if confidence > 60:
        st.markdown(f'<p class="confidence-high">✅ Confidence: {confidence:.0f}%</p>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="confidence-low">⚠️ Confidence: {confidence:.0f}% (Please try rephrasing)</p>', 
                   unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display alternative answers if confidence is low
    if len(indices[0]) > 1:
        with st.expander("📋 Alternative Answers (Top 2 more results)"):
            for i in range(1, min(3, len(indices[0]))):
                alt_idx = indices[0][i]
                alt_distance = distances[0][i]
                alt_confidence = max(0, 100 - (alt_distance * 50))
                
                st.markdown(f"**Result {i+1}** (Confidence: {alt_confidence:.0f}%)")
                st.markdown(lines[alt_idx])
                st.markdown("---")
    
    # Display instructions for next steps
    st.info("💬 Try asking another question or rephrase your question for better results!")

elif user_question and not search_button:
    st.info("👆 Click the Search button to get an answer")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.9em;">
    Built with ❤️ using Streamlit, SentenceTransformers, and FAISS | 
    <a href="https://www.lambtoncollege.ca/" target="_blank">Lambton College</a>
    </div>
""", unsafe_allow_html=True)
