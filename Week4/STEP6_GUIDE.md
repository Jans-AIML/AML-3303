# Step 6: Interactive Chatbot - Deployment Guide

## Overview

You have two options to run the chatbot:

### Option 1: Interactive Notebook Interface (Recommended for Now) ✅
- Runs directly in the Jupyter notebook
- No Terminal commands needed  
- Works with ipywidgets

**Status**: This is already working in your notebook! Click the "🔍 Search" button to test it.

### Option 2: Standalone Streamlit App (Recommended for Deployment)
- Professional web application interface
- Shareable with others
- Better UI/UX
- Requires terminal commands

---

## Using the Notebook Interface

The interactive chatbot in your notebook is ready to use. Simply:

1. **Execute the Step 6 cell** (if you haven't already)
2. **Type your question** in the text input box
3. **Click the "🔍 Search" button**
4. **View the results** with confidence scores and alternative answers

### Example Questions to Try:
- "What are the admission requirements?"
- "How much does tuition cost?"
- "When does the library open?"
- "Does Lambton offer online programs?"
- "What student services are available?"

---

## Running the Streamlit App (Production Option)

If you want to deploy the professional web interface, follow these steps:

### Step 1: Open Terminal

**On Windows:**
- Press `Ctrl + Shift + Backtick` in VS Code, or
- Terminal → New Terminal

**On Mac/Linux:**
- Press `` Ctrl + ` `` in VS Code

### Step 2: Navigate to Week4 Folder

```bash
cd Week4
```

or from the root:

```bash
cd vscode-vfs://github/Jans-AIML/AML-3303/Week4
```

### Step 3: Run Streamlit App

```bash
streamlit run chatbot_app.py
```

### Step 4: Access the App

A browser window will open automatically showing your chatbot at:
```
http://localhost:8501
```

If it doesn't open automatically, visit the URL in your browser.

### Step 5: Stop the App

In the terminal, press `Ctrl + C` to stop the Streamlit app.

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**: Install Streamlit

```bash
pip install streamlit
```

Or if using conda:

```bash
conda install streamlit
```

### Problem: "ModuleNotFoundError: No module named 'week4_rag_data'"

**Solution**: Make sure you're in the Week4 directory when running the command

```bash
cd Week4
streamlit run chatbot_app.py
```

### Problem: "Port 8501 already in use"

**Solution**: Either:
1. Run the app on a different port:
   ```bash
   streamlit run chatbot_app.py --server.port 8502
   ```
2. Or stop other Streamlit instances first

### Problem: "SentenceTransformers model download takes too long"

**Solution**: The first run will download the `all-MiniLM-L6-v2` model (~100MB), which may take a few minutes. Subsequent runs will use the cached model.

---

## Features of the Streamlit App

### Main Features
- **Natural language questions** - Ask questions in any phrasing
- **Semantic search** - Finds semantically similar answers (not just keyword matching)
- **Confidence scores** - Shows how confident the system is in each answer
- **Alternative results** - Shows top 3 results if you want to explore more
- **Responsive UI** - Works on desktop and mobile browsers
- **Sidebar info** - FAQ categories and usage tips

### UI Elements

```
┌─────────────────────────────────────────┐
│   🎓 Campus FAQ Chatbot                 │
│   Ask questions about Lambton College   │
├─────────────────────────────────────────┤
│ [Your question here?          ] [Search]│
├─────────────────────────────────────────┤
│ Your Question: your question            │
│ ┌───────────────────────────────────┐  │
│ │ 🎯 Best Match Answer              │  │
│ │ Q&A content here...               │  │
│ │ ✅ Confidence: 85%                │  │
│ └───────────────────────────────────┘  │
│ 📋 Alternative Answers (expandable)    │
└─────────────────────────────────────────┘
```

---

## Files Generated

### In Week4 folder:

1. **`chatbot_app.py`** - Standalone Streamlit application
2. **`week4_rag_data.py`** - FAQ data module (imported by chatbot_app.py)
3. **`RAG.ipynb`** - Your notebook with interactive chatbot (Step 6)

---

## How the RAG Pipeline Works (For Reference)

```
User Question
    ↓
[Embed] → Convert to 384-dimensional vector
    ↓
[Search in FAISS] → Find 3 nearest neighbors
    ↓
[Rank by Distance] → Calculate confidence scores
    ↓
[Display Results] → Show best answer + alternatives
```

**Key Metrics:**
- Embedding Model: `all-MiniLM-L6-v2` (fast and accurate)
- Embedding Dimension: 384 features
- Index Type: FAISS IndexFlatL2 (Euclidean distance)
- FAQ Dataset: 30 Q&A pairs (can be expanded)

---

## Next Steps

### To Improve the Chatbot:

1. **Add More FAQs** - Edit `week4_rag_data.py` to add more Q&A pairs
2. **Verify Data** - Cross-check with https://www.lambtoncollege.ca/
3. **Test Variations** - Try different question phrasings
4. **Monitor Accuracy** - Track which questions fail
5. **Expand Categories** - Add more specific program information

### To Extend Functionality:

1. **Add feedback** - Let users rate answer quality
2. **Add follow-up** - Enable context-aware follow-up questions
3. **Add logging** - Track all queries for analysis
4. **Add reranking** - Use a more sophisticated model to rerank results
5. **Add generative layer** - Use an LLM to generate natural responses instead of exact FAQ text

---

## Quick Reference Commands

### Run the notebook chatbot (Recommended for learning)
```
Already available in Step 6 of your notebook - just execute the cell!
```

### Run the Streamlit app
```bash
cd Week4
streamlit run chatbot_app.py
```

### Install dependencies
```bash
pip install streamlit sentence-transformers faiss-cpu numpy
```

### Check installed packages
```bash
pip list | grep -E "streamlit|sentence-transformers|faiss"
```

---

## Questions for Reflection (Step 7)

1. **How does embedding capture meaning?** 
   - Why can "What time is the library open?" match "When are library hours?"

2. **What happens with out-of-scope questions?**
   - Try asking something not in the FAQ - observe the confidence score

3. **Why show alternative results?**
   - Can inform user that multiple relevant answers exist

4. **How to improve for production?**
   - Better ranking, filtering, or hybrid search approaches

---

**Need Help?** Check the README.md or reach out to your instructor!
