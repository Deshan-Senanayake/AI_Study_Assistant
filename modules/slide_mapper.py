from rapidfuzz import fuzz

def find_matching_slide(query, docs):
    query_slide = extract_slide_number(query)
    if query_slide is None:
        return None

    best_match = None
    best_score = 0
    for doc in docs:
        slide_meta = doc.metadata.get("slide", "")
        score = fuzz.partial_ratio(slide_meta.lower(), f"slide {query_slide}".lower())
        if score > best_score:
            best_score = score
            best_match = doc

    return best_match

def extract_slide_number(text):
    import re
    match = re.search(r'slide\s*(\d+)', text.lower())
    return int(match.group(1)) if match else None
