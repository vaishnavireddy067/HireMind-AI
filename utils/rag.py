# 🤖 RAG (Retrieval-Augmented Generation) using Groq AI
# This enables chatting with candidate resumes using AI

from groq import Groq
import os

# Store resume texts for RAG
resume_cache = {}

def create_resume_vector_db(text, candidate_id=None):
    """
    Stores resume text for RAG querying.
    Instead of complex embeddings, we use Groq's context understanding.
    """
    # Store the resume text (we'll pass it as context to Groq)
    return {"text": text, "chunks": chunk_text(text)}

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks for better context retrieval.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def find_relevant_chunks(chunks, query, top_k=3):
    """
    Simple keyword-based relevance scoring.
    Returns most relevant chunks based on query keywords.
    """
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        # Score = number of query words found in chunk
        score = sum(1 for word in query_words if word in chunk_lower)
        scored_chunks.append((score, chunk))
    
    # Sort by score descending and return top_k
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:top_k]]

def query_resume(vector_store, query):
    """
    🤖 RAG-powered resume Q&A using Groq AI.
    Retrieves relevant context and generates AI answer.
    """
    if not vector_store:
        return "❌ Resume data not available."
    
    try:
        # Get relevant chunks from resume
        chunks = vector_store.get("chunks", [])
        
        if not chunks:
            return "❌ No resume content found."
        
        # Custom relevance logic to handle "generic" queries (e.g. "is he good?")
        query_words = set(query.lower().split())
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            score = sum(1 for word in query_words if word in chunk.lower())
            # Give slight boost to early chunks (Summary/Intro) for generic queries
            if i < 2: score += 0.5 
            scored_chunks.append((score, chunk))
            
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # If best match is weak (low score), prefer the beginning of the resume
        if scored_chunks[0][0] < 1:
            best_chunks = chunks[:4]
        else:
            best_chunks = [chunk for score, chunk in scored_chunks[:3]]
            
        context = "\n\n".join(best_chunks)
        
        # Check for API Key in environment
        api_key = os.environ.get("GROQ_API_KEY")
        
        if api_key:
            # Use Groq for smart answers
            client = Groq(api_key=api_key)
            
            prompt = f"""You are an expert HR AI Assistant analyzing a candidate's resume.
            
📄 **Resume Context:**
{context}

❓ **User Question:** {query}

**Instructions:**
- If the question is generic (e.g., "is he good?", "summarize", "thoughts?"), provide a professional underlying summary of the candidate's strengths and weaknesses based on the resume.
- If the question is specific (e.g., "Python experience?"), answer directly with facts from the resume.
- Be helpful, concise, and professional.
- If the resume is missing info, say "The resume doesn't mention X, but..."
"""

            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
            )
            return f"🤖 **AI Answer:**\n\n{response.choices[0].message.content}"
        
        else:
            # Fallback: Return relevant context without AI processing
            return f"""🔍 **Relevant Resume Sections:**

{context}

---
💡 *Tip: Set `GROQ_API_KEY` environment variable for AI-powered answers!*"""
    
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
