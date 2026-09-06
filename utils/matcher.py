from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def rank_resumes(job_description, resume_text):
    """
    Ranks a resume against a job description using TF-IDF and Cosine Similarity.
    Returns a score (0-100) and a summary string.
    """
    documents = [job_description, resume_text]
    
    # Simple TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    
    # Calculate Cosine Similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    # Get the score
    match_percentage = round(similarity_matrix[0][0] * 100, 2)
    
    # Skill Gap Analysis (Basic)
    # Get feature names (words) from the vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Get the non-zero elements for JD (index 0)
    jd_vector = tfidf_matrix[0]
    jd_indices = jd_vector.nonzero()[1]
    jd_keywords = {feature_names[i] for i in jd_indices}
    
    # Get the non-zero elements for Resume (index 1)
    resume_vector = tfidf_matrix[1]
    resume_indices = resume_vector.nonzero()[1]
    resume_keywords = {feature_names[i] for i in resume_indices}
    
    # Find missing keywords (Present in JD but not in Resume)
    missing_keywords = list(jd_keywords - resume_keywords)
    
    # Find matching keywords (Intersection) - The "Why" behind the score
    common_keywords = list(jd_keywords.intersection(resume_keywords))
    
    return match_percentage, missing_keywords, common_keywords
