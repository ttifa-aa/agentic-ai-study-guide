"""
Text processing utilities for academic content analysis.
Includes topic extraction, keyword detection, and content categorization.
"""

import re  # regular expressions for pattern matching in text extraction
from typing import List, Dict, Set, Optional, Tuple  # type hints for function signatures
from collections import Counter  # for counting word frequencies in topic extraction
import spacy  # advanced nlp library for named entity recognition and parsing
import nltk  # natural language toolkit for text processing and tokenization
from nltk.corpus import stopwords  # common words to filter out during analysis
from nltk.tokenize import sent_tokenize, word_tokenize  # sentence and word tokenizers

# Download required NLTK data - ensures necessary corpora are available
try:
    nltk.data.find('tokenizers/punkt')
    # checks if punkt tokenizer is already downloaded
except LookupError:
    nltk.download('punkt')
    # downloads punkt tokenizer for sentence splitting if not present

try:
    nltk.data.find('corpora/stopwords')
    # checks if stopwords corpus is already downloaded
except LookupError:
    nltk.download('stopwords')
    # downloads stopwords corpus for filtering common words if not present

# Load spaCy model for academic text processing
# the english small model provides good balance of accuracy and performance
try:
    nlp = spacy.load("en_core_web_sm")
    # attempts to load the small english spacy model for nlp tasks
except OSError:
    # If model not available, use basic processing
    # fallback to simpler methods if spacy model isn't installed
    nlp = None
    # sets nlp to none - functions will check this and use fallback methods

# Academic keywords for topic extraction - words that often indicate important concepts
ACADEMIC_KEYWORDS: Set[str] = {
    "definition", "theorem", "algorithm", "principle", "concept",
    # core academic vocabulary indicating formal knowledge
    "method", "technique", "approach", "framework", "model",
    # methodological terms that signal structured thinking
    "theory", "application", "example", "exercise", "problem",
    # pedagogical terms commonly found in educational materials
    "solution", "analysis", "synthesis", "evaluation", "implementation",
    # bloom's taxonomy terms indicating higher-order thinking
    "architecture", "design", "pattern", "paradigm", "methodology"
    # technical terms used across multiple disciplines
}

# Common academic stopwords to filter - words that appear frequently but add little meaning
# combines standard english stopwords with academic-specific filler words
ACADEMIC_STOPWORDS: Set[str] = set(stopwords.words('english')).union({
    "figure", "table", "chapter", "section", "page", "example",
    # document structure words that don't indicate topic content
    "therefore", "thus", "hence", "however", "moreover"
    # transition words that don't carry topic-specific meaning
})


def extract_topics(text: str, max_topics: int = 10) -> List[str]:
    """
    Extract main academic topics from document content.
    
    Args:
        text (str): The document text to analyze
        max_topics (int): Maximum number of topics to return
    
    Returns:
        List[str]: List of extracted topic phrases
    """
    topics: List[str] = []  # initialize empty list to store extracted topic phrases
    
    # Method 1: Extract from sentences containing academic keywords
    # academic keywords help identify sentences that likely contain topic information
    sentences = sent_tokenize(text)  # split text into individual sentences for analysis
    for sentence in sentences:  # iterate through each sentence to find topic indicators
        if any(keyword in sentence.lower() for keyword in ACADEMIC_KEYWORDS):
            # check if sentence contains any academic keywords (case-insensitive)
            # if spacy is available, use it for more accurate noun phrase extraction
            if nlp:
                doc = nlp(sentence[:500])  # limit length to first 500 chars for performance
                # extract noun phrases - these often represent key concepts and topics
                noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks 
                               if 3 <= len(chunk.text.split()) <= 6]
                # keep phrases between 3-6 words - too short lacks context, too long is unwieldy
                topics.extend(noun_phrases)  # add extracted noun phrases to topics list
            else:
                # Fallback: Extract capitalized phrases when spacy isn't available
                # capitalized phrases often indicate proper nouns and technical terms
                capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
                # regex matches words starting with capital letters, possibly multiple words
                topics.extend(capitalized)  # add capitalized phrases to topics list
    
    # Method 2: Extract section headings (ALL CAPS or Title Case)
    # headings often contain the most important topic information
    heading_pattern = r'^[A-Z][A-Z\s]{5,}$|^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}$'
    # pattern matches either all-caps headings or title case phrases of 2-6 words
    lines = text.split('\n')  # split text into lines to check each for heading patterns
    for line in lines:  # iterate through each line
        if re.match(heading_pattern, line.strip()):  # check if line matches heading pattern
            topics.append(line.strip())  # add heading text to topics list
    
    # Clean and deduplicate - normalize topics and remove duplicates
    cleaned_topics = _clean_topic_list(topics)  # call helper function to clean topic strings
    
    # Count frequencies and return top N
    topic_counts = Counter(cleaned_topics)  # count occurrences of each cleaned topic
    return [topic for topic, _ in topic_counts.most_common(max_topics)]
    # return the most frequent topics up to max_topics limit


def _clean_topic_list(topics: List[str]) -> List[str]:
    """
    Clean and normalize topic phrases.
    
    Args:
        topics (List[str]): Raw topic phrases
    
    Returns:
        List[str]: Cleaned topic phrases
    """
    cleaned = []  # initialize empty list for cleaned topics
    
    for topic in topics:  # process each topic phrase individually
        # Remove special characters and extra whitespace
        topic = re.sub(r'[^\w\s-]', '', topic)  # keep only word chars, spaces, and hyphens
        topic = ' '.join(topic.split())  # normalize whitespace to single spaces
        
        # Filter out stopwords-only phrases - these carry no meaningful information
        words = topic.lower().split()  # convert to lowercase and split into words
        if words and not all(word in ACADEMIC_STOPWORDS for word in words):
            # keep only if there's at least one meaningful word (not all stopwords)
            # Keep phrases between 2-6 words for optimal readability and specificity
            if 2 <= len(words) <= 6:
                cleaned.append(topic)  # add valid topic to cleaned list
    
    return cleaned  # return the list of cleaned and filtered topics


def detect_content_type(text: str) -> str:
    """
    Automatically detect the type of academic content.
    
    Args:
        text (str): The document text
    
    Returns:
        str: Detected content type
    """
    text_lower = text.lower()  # convert to lowercase for case-insensitive matching
    
    # Check for code/algorithm content - common in cs and programming materials
    code_indicators = ["def ", "function", "class ", "import ", "#include", "algorithm"]
    # keywords that strongly indicate programming or algorithmic content
    if any(indicator in text_lower for indicator in code_indicators):
        return "technical_content"  # return technical content for code-heavy materials
    
    # Check for mathematical content - equations, theorems, proofs
    math_indicators = ["equation", "formula", "theorem", "proof", "lemma", "corollary"]
    # mathematical terminology indicating formal mathematical content
    math_score = sum(1 for ind in math_indicators if ind in text_lower)
    # count how many math indicators are present
    if math_score >= 2:  # require at least 2 indicators for confidence
        return "mathematical_content"  # return mathematical content type
    
    # Check for exam content - questions, marks, problem-solving
    exam_indicators = ["question", "marks", "answer", "explain", "solve", "calculate"]
    # words commonly found in examination papers and problem sets
    exam_score = sum(1 for ind in exam_indicators if ind in text_lower)
    # count exam-related indicators
    if exam_score >= 3:  # require at least 3 indicators for exam content confidence
        return "exam_content"  # return exam content type
    
    # Check for practical/lab content - experiments and procedures
    lab_indicators = ["experiment", "procedure", "apparatus", "observation", "result"]
    # terms associated with laboratory manuals and practical sessions
    if any(ind in text_lower for ind in lab_indicators):
        return "practical_content"  # return practical content type
    
    # Default to theoretical content if no specific type detected
    return "theoretical_content"  # fallback for general academic theory materials


def extract_key_terms(text: str, max_terms: int = 15) -> List[Tuple[str, float]]:
    """
    Extract key technical terms from academic text with relevance scores.
    
    Args:
        text (str): The document text
        max_terms (int): Maximum number of terms to return
    
    Returns:
        List[Tuple[str, float]]: List of (term, relevance_score) tuples
    """
    if nlp:  # use spacy for more accurate term extraction if available
        doc = nlp(text[:5000])  # limit to first 5000 chars for performance
        
        # Extract named entities - proper nouns like people, organizations, places
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "GPE", "PERSON"]]
        # keep only relevant entity types - organizations, products, locations, people
        
        # Extract technical terms (nouns with specific patterns)
        technical_terms = []  # list to store identified technical terms
        for token in doc:  # examine each token (word) in the document
            if token.pos_ == "NOUN" and len(token.text) > 3:
                # focus on nouns longer than 3 characters (filter out short/trivial words)
                compound = token.text  # start with the noun itself
                if token.dep_ == "compound":
                    # Get the full compound phrase when noun is part of a compound
                    # compound dependency indicates this noun modifies another noun
                    phrase = ' '.join([t.text for t in token.head.subtree 
                                      if t.dep_ in ["compound", "amod"]])
                    # collect all words in the compound noun phrase
                    if phrase:
                        compound = f"{phrase} {token.head.text}"
                        # combine modifiers with the head noun for complete term
                technical_terms.append(compound)  # add to technical terms list
        
        # Combine and score terms - entities and technical terms together
        all_terms = entities + technical_terms  # merge both sources of terms
        term_counts = Counter(all_terms)  # count frequency of each term
        
        # Calculate relevance scores (simplified TF-IDF like scoring)
        total_terms = len(all_terms)  # total number of terms extracted
        scored_terms = [(term, count/total_terms) for term, count in term_counts.items()]
        # score is term frequency divided by total terms - simple tf scoring
        
        return sorted(scored_terms, key=lambda x: x[1], reverse=True)[:max_terms]
        # return top scoring terms up to max_terms limit
    
    # Fallback: Simple word frequency analysis when spacy not available
    words = word_tokenize(text.lower())  # tokenize and lowercase the text
    words = [w for w in words if w.isalpha() and len(w) > 3 and w not in ACADEMIC_STOPWORDS]
    # filter for alphabetic words longer than 3 chars that aren't stopwords
    word_counts = Counter(words)  # count word frequencies
    total = len(words)  # total number of filtered words
    
    return [(word, count/total) for word, count in word_counts.most_common(max_terms)]
    # return top words with frequency scores


def chunk_by_semantic_boundaries(text: str, chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks based on semantic boundaries (paragraphs, sections).
    better than simple character splitting as it preserves meaning boundaries.
    
    Args:
        text (str): The text to chunk
        chunk_size (int): Target chunk size in characters
    
    Returns:
        List[str]: List of text chunks
    """
    chunks = []  # initialize empty list for resulting chunks
    current_chunk = ""  # current chunk being built
    
    # Split by double newlines (paragraph boundaries)
    # paragraphs are natural semantic units in academic writing
    paragraphs = text.split('\n\n')  # double newline indicates paragraph break
    
    for para in paragraphs:  # process each paragraph
        if len(current_chunk) + len(para) < chunk_size:
            # if adding this paragraph keeps chunk under size limit
            current_chunk += para + "\n\n"  # add paragraph with spacing
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())  # save completed chunk
            
            # If paragraph is still too large, split by sentences
            if len(para) > chunk_size:
                sentences = sent_tokenize(para)  # split large paragraph into sentences
                current_chunk = ""  # reset for new chunk
                for sent in sentences:  # process each sentence
                    if len(current_chunk) + len(sent) < chunk_size:
                        current_chunk += sent + " "  # add sentence to current chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())  # save completed chunk
                        current_chunk = sent + " "  # start new chunk with current sentence
            else:
                current_chunk = para + "\n\n"  # start new chunk with this paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())  # add the final chunk
    
    return chunks  # return list of semantically-aware chunks