import re
import random

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        nlp = spacy.blank("en") # Fallback to blank model if download failed
except ImportError:
    nlp = None

import os
from groq import Groq

def semantic_match(jd, resume_text):
    """
    Computes semantic match score using Groq AI (Llama 3) for high accuracy.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            client = Groq(api_key=api_key)
            prompt = f"""
            Act as an expert ATS (Applicant Tracking System).
            Evaluate the match between the Resume and Job Description (JD).
            
            Give a match score from 0 to 100 based on skills, experience, and relevance.
            Be strict but fair. Do NOT use keyword matching only; use semantic understanding.
            
            JD: {jd[:1000]}...
            Resume: {resume_text[:1000]}...
            
            Return ONLY the number (e.g. 85). Nothing else.
            """
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
            )
            score_str = chat_completion.choices[0].message.content.strip()
            # Extract number
            import re
            match = re.search(r'\d+', score_str)
            if match:
                return float(match.group())
        except Exception as e:
            print(f"Groq Scoring Error: {e}")
            pass
            
    # Fallback: Jaccard Similarity (Better than strict TF-IDF for short text)
    jd_words = set(jd.lower().split())
    resume_words = set(resume_text.lower().split())
    if not jd_words: return 0.0
    common = jd_words.intersection(resume_words)
    return round(len(common) / len(jd_words) * 100, 2)

def redact_pii(text):
    """
    Redacts Names (PERSON), Emails, and Phones using Spacy + Regex.
    """
    redacted_text = text
    
    # 1. Regex for Email/Phone
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
    
    redacted_text = re.sub(email_pattern, "[EMAIL REDACTED]", redacted_text)
    redacted_text = re.sub(phone_pattern, "[PHONE REDACTED]", redacted_text)
    
    # 2. Spacy for Names (only if loaded)
    if nlp:
        doc = nlp(redacted_text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                redacted_text = redacted_text.replace(ent.text, "[NAME REDACTED]")
                
    return redacted_text

def correct_bias(text):
    """
    Simple bias checking for JDs.
    """
    biased_words = {
        "ninja": "developer",
        "rockstar": "expert",
        "he": "they",
        "she": "they",
        "guys": "team",
        "young": "energetic"
    }
    suggestions = []
    lower_text = text.lower()
    for word, replacement in biased_words.items():
        if f" {word} " in f" {lower_text} ":
            suggestions.append(f"Replace '{word}' with '{replacement}'")
            
    return suggestions

def generate_interview_questions(resume_text):
    """
    Generates relevant interview questions based on keywords found in the text.
    """
    questions = []
    
    # Keyword-based Question Bank
    bank = {
        "python": [
            "Explain the difference between list and tuple.",
            "How does memory management work in Python?",
            "What are Python decorators?"
        ],
        "machine learning": [
            "What is overfitting and how do you prevent it?",
            "Explain the bias-variance tradeoff.",
            "How do you select important features?"
        ],
        "sql": [
            "Explain the difference between INNER JOIN and LEFT JOIN.",
            "What is a primary key vs foreign key?",
            "How do you optimize a slow query?"
        ],
        "react": [
            "What is the Virtual DOM?",
            "Explain the useEffect hook.",
            "State vs Props?"
        ],
        "django": [
            "Explain the MVT architecture.",
            "How does Django Middleware work?"
        ],
        "aws": [
            "Explain EC2 vs Lambda.",
            "What is S3 used for?"
        ],
        "docker": [
            "Difference between Image and Container?",
            "Explain Docker Compose."
        ]
    }
    
    resume_lower = resume_text.lower()
    for skill, qs in bank.items():
        if skill in resume_lower:
            questions.extend(random.sample(qs, 1)) # Pick 1 random question per skill
            
    if not questions:
        questions.append("Tell me about the most challenging project you've worked on.")
        
    return list(set(questions)) # Dedupe if any

def predict_hiring_chance(score):
    """
    Predicts likelihood of hiring based on score.
    Simple rule-based logic (can be replaced by ML model).
    """
    if score >= 80:
        return "High (Recommended)", "green"
    elif score >= 50:
        return "Medium (Potential)", "orange"
    else:
        return "Low (Reject)", "red"

def check_authenticity(text):
    """
    Detects potential red flags in resume content.
    """
    flags = []
    
    # 1. Keyword Stuffing Check (Heuristic: Duplicate words ratio)
    words = text.lower().split()
    unique_words = set(words)
    if len(words) > 500 and (len(unique_words) / len(words)) < 0.3:
        flags.append("Possible Keyword Stuffing Detected")
        
    # 2. Length Check
    if len(words) < 50:
        flags.append("Resume too short (Likely fake or incomplete)")
        
    # 3. Contact Info Missing check happens elsewhere, but valid check here too
    if "@" not in text:
         flags.append("Missing Email Address")
         
    return flags if flags else ["✅ Authentic"]

def generate_email(candidate_name, job_role, score):
    """
    Generates a personalized email draft.
    """
    if score >= 60:
        subject = f"Interview Invitation for {job_role} Role - {candidate_name}"
        body = f"""Dear {candidate_name},

We carefully reviewed your application for the {job_role} position and were impressed by your profile (Match Score: {score}%).

We would love to invite you for an interview to discuss your experience further.

Please let us know your availability for a call this week.

Best Regards,
Hiring Team"""
    else:
        subject = f"Update regarding your application for {job_role}"
        body = f"""Dear {candidate_name},

Thank you for applying to the {job_role} position.

After careful review, we have decided to move forward with other candidates who more closely match our current requirements.

We will keep your resume on file for future openings.

Best Regards,
Hiring Team"""
        
    return subject, body
