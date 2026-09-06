from deep_translator import GoogleTranslator

def translate_resume(text):
    """
    Detects non-English text and translates it to English using Deep Translator.
    Limit to first 3000 chars to avoid timeouts.
    """
    try:
        # Simple heuristic: if > 50% of the first 100 chars are ASCII, assume English
        sample = text[:100]
        if len(sample.encode('utf-8')) == len(sample):
            return text 

        # Otherwise, translate
        translator = GoogleTranslator(source='auto', target='en')
        # Split into chunks of 4500 chars (limit of Google Translate)
        chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
        translated_text = ""
        for chunk in chunks:
            translated_text += translator.translate(chunk) + " "
        return translated_text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text # Return original if fails
