import re

def extract_contact_info(text):
    """
    Extracts email and phone number from text using Regex.
    """
    # Email Regex
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email = re.findall(email_pattern, text)
    
    # Phone Regex (Simple version, supports various formats)
    # Matches: +1-123-456-7890, (123) 456-7890, 123 456 7890, etc.
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
    phone = re.findall(phone_pattern, text)
    
    # Return first match or "Not Found"
    return {
        "email": email[0] if email else "Not Found",
        "phone": "".join(phone[0]) if phone else "Not Found"
    }

def extract_name(text):
    """
    Extracts candidate name using heuristic (First line usually).
    """
    lines = text.split('\n')
    for line in lines[:10]: # Check first 10 lines
        clean_line = line.strip()
        if len(clean_line) > 3 and len(clean_line) < 50:
            # Check if it's not a label like "Resume", "CV", "Page 1"
            lower_line = clean_line.lower()
            if not any(keyword in lower_line for keyword in ["resume", "curriculum", "bio-data", "cv", "page"]):
                 # Check if it looks like a name (mostly letters)
                 if re.match(r'^[a-zA-Z\s\.\'-]+$', clean_line):
                     return clean_line.title()
    return "Unknown Candidate"
