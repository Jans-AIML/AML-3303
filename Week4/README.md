# Week 4: Retrieval-Augmented Generation (RAG) for Campus FAQ Chatbot

## Overview

This week's activity focuses on **Retrieval-Augmented Generation (RAG)**, a powerful technique that combines information retrieval with generative AI. Students will build a campus FAQ chatbot that retrieves relevant information and generates responses using vector embeddings and semantic search.

## Activity Description

### RAG.ipynb - Campus FAQ Chatbot Project

This Jupyter notebook is an **in-class learning activity** that teaches students how to build a functional RAG system from scratch, using real college data (Lambton College).

#### What You'll Build

A question-answering chatbot that:
1. **Retrieves** relevant FAQ entries based on user questions
2. **Ranks** results by semantic similarity
3. **Returns** the most relevant answer from the knowledge base

#### Project Steps

| Step | Task | Objective |
|------|------|-----------|
| **Step 0** | Setup & Dependencies | Install required ML and NLP libraries |
| **Step 1** | Prepare FAQ Data | Create a structured dataset of Q&A pairs |
| **Step 2** | Text Chunking | Split documents into retrievable chunks |
| **Step 3** | Create Embeddings | Convert text to vector representations |
| **Step 4** | Build Vector Index | Store embeddings in FAISS database |
| **Step 5** | Query & Retrieve | Search for relevant answers |
| **Step 6** | Interactive UI | Deploy with Streamlit chatbot interface |
| **Step 7** | Reflection | Analyze system performance & improvements |

## Data Preparation

### Dataset Structure

Two data sources are provided for this project:

#### 1. In-Notebook FAQ Dataset (`faq_data`)
- **Format**: Python dictionary organized by category
- **Categories**: Admissions, Programs, Campus Facilities, Financial Aid, Student Services, Academic Calendar, Career Services, Student Life, Contact
- **Size**: 28 Q&A pairs (~2,800 words)
- **Advantage**: Lightweight, no file I/O needed

**Categories covered:**
- 🎓 **Admissions** - Application requirements, deadlines, international students
- 📚 **Programs** - Offered programs, duration, online options
- 🏫 **Campus Facilities** - Library hours, residences, parking, recreation
- 💰 **Tuition & Financial Aid** - Cost estimates, scholarships, OSAP eligibility
- 🤝 **Student Services** - Advising, counseling, accessibility support
- 📅 **Academic Calendar** - Term dates, application deadlines, semester structure
- 🎯 **Career Services** - Job placement, co-op programs, employer partnerships
- 🎉 **Student Life** - Clubs, sports, events, campus activities
- 📞 **Contact & Locations** - Main address, phone, email

#### 2. CSV File (`lambton_college_faq.csv`)
- **Format**: Structured CSV with metadata
- **Columns**: `category`, `question`, `answer`, `source`
- **Advantage**: Easy to export, version control, manual verification
- **Use**: Loading with pandas for analysis or external tools

### RAG Pipeline Component Functions

The notebook includes cells that demonstrate:

```python
# 1. Text Preparation
document_chunks = [...]  # Individual retrievable units

# 2. Embedding Generation (using SentenceTransformer)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(document_chunks)

# 3. Vector Database (using FAISS)
import faiss
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 4. Semantic Search
user_question = "What's the tuition cost?"
q_embedding = model.encode([user_question])
distances, indices = index.search(q_embedding, k=3)  # Top 3 results

# 5. Result Retrieval
relevant_answer = document_chunks[indices[0][0]]
```

## How RAG Works in This Project

### The RAG Pipeline

```
User Question
       ↓
[Embed Question] → Convert to vector (384-dimensional)
       ↓
[Search Index] → Find K nearest neighbors in FAISS
       ↓
[Rank Results] → Sort by semantic similarity
       ↓
[Return Answer] → Display most relevant FAQ entry
```

### Why RAG is Better Than Traditional Chatbots

| Aspect | Traditional Rule-Based | Keyword Search | RAG (Semantic Search) |
|--------|----------------------|-----------------|----------------------|
| Understanding | Pattern matching | Keywords only | Semantic meaning |
| Flexibility | Rigid rules | Brittle queries | Flexible & adaptive |
| Example Query | "library hours?" | Works | Works | Works ✓ |
| Example Query | "when is the library open?" | May fail | May fail | Works ✓ |
| Example Query | "Where can I study?" | May fail | May fail | Understands intent ✓ |
| Scalability | Limited | Limited | Scales with data |

## Technical Concepts

### Vector Embeddings
- **What**: Numerical representation of text meaning
- **Why**: Enable semantic similarity comparison
- **How**: Pre-trained model (all-MiniLM-L6-v2) converts text to 384-dimensional vectors
- **Benefit**: Similar questions → similar vectors → easy to find related FAQs

### FAISS Index
- **What**: Facebook AI Similarity Search - fast vector search library
- **Why**: Enables quick nearest-neighbor lookup in millions of vectors
- **Implementation**: IndexFlatL2 uses Euclidean distance
- **Performance**: O(n) search, can be optimized with IndexIVFFlat for large datasets

### Chunking Strategy
- **What**: Split documents into meaningful pieces
- **Why**: Better retrieval granularity, manage context windows
- **This project**: Each Q&A pair is a chunk
- **Best practice**: Overlapping chunks for comprehensive coverage

## Data Quality & Verification

### ✅ Current Data Source
- Based on typical Lambton College public information
- Categorized for easy retrieval
- Includes realistic tuition ranges and services

### 🔍 How to Verify & Improve

1. **Check the Official Website**
   - Visit https://www.lambtoncollege.ca/
   - Verify key facts (tuition, deadlines, contact info)
   - Update `lambton_college_faq.csv` with actual data

2. **Add More FAQs**
   - Mining website content for additional Q&A pairs
   - Including specific program requirements
   - Adding international student-specific information
   - Covering career outcomes and alumni stories

3. **Create Variations**
   - For each answer, create multiple question phrasings
   - Example: "When does the library open?" vs "What are library hours?" vs "Is the library open at 9 AM?"
   - Improves retrieval accuracy for different user queries

4. **Add Context**
   - Include hyperlinks to detailed pages
   - Add timestamps for time-sensitive information
   - Include contact info for specific departments

## ML & AI Concepts Demonstrated

### Concept 1: Information Retrieval
- How to find relevant documents efficiently
- Vector-based similarity search
- Ranking and reranking results

### Concept 2: Transfer Learning
- Using pre-trained embeddings model (not training from scratch)
- Saves computational resources
- Enables semantic understanding with minimal data

### Concept 3: Vector Databases
- Designing efficient data structures for similarity search
- Index types and trade-offs (speed vs accuracy)
- Scaling search to millions of documents

### Concept 4: RAG Pipeline Architecture
- Modular design: separate retrieval & generation stages
- Evaluation metrics: precision, recall, ranking quality
- Error analysis: retrieving wrong documents

## Learning Outcomes

By completing this activity, you will:

✅ Understand the RAG (Retrieval-Augmented Generation) architecture  
✅ Learn how to create document embeddings using transformer models  
✅ Build and query a vector database using FAISS  
✅ Design effective chunking strategies for documents  
✅ Implement semantic search for Q&A systems  
✅ Deploy an interactive chatbot with Streamlit  
✅ Evaluate and improve retrieval quality  

## Practical Applications

RAG systems are used in production by:

- **ChatGPT Plugins**: Retrieval-augmented generation for current information
- **Corporate Support**: Employee/customer FAQ automaton
- **Medical Systems**: Doctor decision support with retrieval-based context
- **Legal Tech**: Contract analysis and case law retrieval
- **E-commerce**: Product recommendation and support ticket automation
- **Education**: Student learning assistants and tutoring systems

## Improvements & Extensions

### For Better Retrieval
1. **Multi-hop Search**: Find answers that require combining multiple FAQ entries
2. **Query Expansion**: Generate related queries to improve recall
3. **Reranking**: Use a more sophisticated model to rerank top-K results
4. **Hybrid Search**: Combine keyword search with semantic search

### For Better Responses
1. **Generative Layer**: Use an LLM to generate natural responses instead of retrieving exact FAQ text
2. **Conversational Memory**: Maintain context across multiple questions
3. **Confidence Scores**: Return reliability metrics with answers
4. **Feedback Loop**: Learn from user feedback to improve responses

### For Production Deployment
1. **Caching**: Cache embeddings for frequently asked questions
2. **Async Processing**: Handle multiple users concurrently
3. **Monitoring**: Track query performance and user satisfaction
4. **Updates**: Efficiently update FAQ database without reindexing

## Resources

- **SentenceTransformers**: https://www.sbert.net/
- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **Streamlit Tutorial**: https://docs.streamlit.io/
- **RAG Best Practices**: https://huggingface.co/docs/transformers/tasks/rag

## Reflection Questions

Use these questions to guide your learning:

1. **Why does the chatbot work well for some questions but fail for others?**
   - Consider semantic similarity and exact question matching

2. **What happens when you ask a question completely outside the FAQ?**
   - How does the system behave? Should it indicate low confidence?

3. **How could you improve this system to handle variations in how people ask questions?**
   - Think about query expansion, multiple phrasings, synonyms

4. **If the FAQ database had 100,000 entries instead of 28, what challenges would arise?**
   - Consider search speed, embedding generation time, memory usage

5. **How would you evaluate if this chatbot is actually helpful to users?**
   - Consider metrics like accuracy, user satisfaction, coverage

---

**Note**: This is an educational project demonstrating RAG principles. The FAQ data should be verified against the actual Lambton College website before deployment in production use.
