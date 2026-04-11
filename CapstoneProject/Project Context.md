Project Context

A growing SaaS company wants to improve its customer support operations by building an internal AI-powered support assistant. The company receives a high volume of repetitive support questions from customers related to product setup, refund policies, troubleshooting, and account usage. Currently, support staff manually search help documents, PDFs, FAQs, and internal guides to answer customer queries, which is time-consuming and inefficient.

The leadership team wants to modernize this workflow using Large Language Models, retrieval-based AI systems, and software automation. The goal is to create a support chatbot that can answer questions based only on uploaded company knowledge documents. The system should demonstrate practical use of modern AI engineering tools, database integration, background processing, and deployment readiness.
Problem Statement

Customer support teams often struggle with scattered documentation, repetitive tickets, and inconsistent response quality. Important information is stored across multiple files and formats, making it difficult to quickly retrieve accurate answers. Manual support handling increases response time, operational cost, and customer frustration.

The company requires an intelligent assistant that can process uploaded documents, retrieve relevant information, and generate professional support responses in real time. The system should also maintain structured records of uploaded files and chat interactions in a database.
Project Objectives

You will design and develop an intelligent AI support system that:

    Accepts and processes support documents such as PDFs, FAQs, and manuals
    Extracts, chunks, and stores document knowledge for retrieval
    Answers user questions using LLM + RAG logic
    Stores document metadata and conversation history in a database
    Demonstrates automation through background document processing

The final solution may run locally for presentation or be deployed on AWS EC2 Free Tier. Students may use cloud LLM APIs or Ollama for local model inference.
2 Week Capstone Timeline
Week 1: Planning and Core System Design

Focus: Turn the idea into a clear technical plan and begin core implementation.

    Problem overview
    User flow diagrams
    System architecture
    Database schema design
    API design
    UI wireframe
    RAG workflow design
    Prompt design strategy

Expected Output: Approved plan and project setup with basic scaffolding completed.
Week 2: Core Development, Testing, and Final Presentation

Focus: Build, stabilize, and present the working system.

    Document upload and parsing
    Chunking and retrieval logic
    Embedding/vector storage setup
    Prompt engineering for support-style answers
    Chatbot question-answer workflow
    Database CRUD operations
    Background automation for document processing
    UI improvements and integration
    Bug fixes and error handling
    Local demo setup or AWS EC2 deployment
    Push code to GitHub
    Preparation of final presentation slides

Expected Output: A working demo-ready AI support chatbot.
Tools to Utilize:

    Draw.io or Lucidchart for architecture diagrams
    Notion or Markdown for tech spec document
    Jupyter Notebook for quick experimentation and prompt testing
    FastAPI for backend API development
    Pydantic for schema validation
    PostgreSQL or Supabase PostgreSQL as the primary database
    ChromaDB, FAISS, or Pinecone for vector retrieval
    LangChain or LlamaIndex for RAG pipeline development
    OpenAI, Gemini, Groq, or Ollama for LLM integration
    PyPDF2, pdfplumber, python-docx, or Unstructured for document parsing
    Streamlit, React, or Next.js for the user interface
    FastAPI BackgroundTasks, APScheduler, or Celery for automation
    AWS EC2, Vercel or any other free tool, Free Tier for optional deployment
    Docker for local containerized setup if preferred
    Ruff for linting

Grading Rubric

Total: 100 marks

System Design and Technical Planning – 20 marks
Quality of architecture, clarity of data flow, database design, and completeness of the tech spec document.

Implementation and Functionality – 30 marks
Functionality of document upload, parsing, retrieval, chatbot workflow, code quality, modularity, and robustness of the application.

AI Integration, RAG and Prompt Engineering – 20 marks
Correctness of the retrieval pipeline, LLM integration, prompt design quality, and relevance/accuracy of generated answers.

Database, Automation and Tool Usage – 15 marks
Effective use of database storage, background automation, and appropriate use of frameworks, libraries, and AI tools.

Presentation, Teamwork and Demo – 15 marks
Clarity and professionalism of the final presentation, quality of live demo, and evidence of teamwork and contribution.
Final Deliverables

    GitHub repository link
    Fully working application with demo
    README file with setup instructions and screenshots
    requirements.txt or package.json files
    .gitignore file
    Tech Spec Document
    Final presentation slides

Optional deployed link or AWS EC2 public access link
Optional features:

Source citations

Analytics dashboard

Authentication

Multi-agent workflow using LangGraph or CrewAI

Embeddable support widget

Conversation summarization

Admin panel
RAG / Support Chatbot Reference

A Retrieval-Augmented Generation support assistant typically relies on the following core components:

1. Document ingestion
The system accepts support knowledge sources such as FAQs, manuals, policy documents, help guides, and product instructions.

2. Text extraction
Uploaded files are converted into machine-readable text so they can be processed by downstream AI components.

3. Chunking
Large documents are split into smaller meaningful sections to improve retrieval quality and reduce prompt size.

4. Embeddings and vector search
Each chunk is converted into a numerical representation so semantically relevant sections can be retrieved when a user asks a question.

5. Retrieval step
The system finds the most relevant document chunks based on the user query.

6. Prompt grounding
The retrieved chunks are inserted into the prompt so the LLM answers using actual document context instead of guessing.

7. Answer generation
The LLM generates a concise, professional support response grounded in the retrieved knowledge.

8. Fallback behavior
If the answer is not present in the uploaded documents, the system should clearly state that the information was not found.

9. Conversation history
Past user questions and assistant responses should be stored in the database for traceability and analytics.

10. Automation
Document processing tasks such as chunking, embedding creation, or indexing should happen automatically after upload or through background jobs.
Ollama Usage Option

Students may use Ollama instead of a paid cloud API.

Suggested local models:

    llama3
    mistral
    phi3
    gemma

This is useful when:

Students want a low-cost solution

The internet/API dependency should be reduced

The final demo is run locally

AWS EC2 Deployment Note

Students may present either:

locally on their own machine

on AWS EC2 Free Tier

or through another cloud platform

If deploying on EC2, students should be mindful of free-tier resource limits. Running larger Ollama models on EC2 may be difficult, so local presentation with Ollama is acceptable. EC2 deployment is optional, not mandatory.
Suggested Core Features

Minimum required features:

    Upload support documents
    Parse document content
    Store document metadata in DB
    Create chunk-based retrieval workflow
    Ask questions through chatbot UI
    Generate answers using LLM + retrieved context
    Save chat history in DB
    Implement one automation/background process
    Present a working live demo