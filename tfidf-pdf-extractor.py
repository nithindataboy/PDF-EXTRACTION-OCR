import pdfplumber
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Extract text from the PDF
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Step 2: Preprocess the text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english') and len(word) > 1]
    return ' '.join(text)

# Step 3: Apply TF-IDF
def apply_tfidf(corpus):
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(corpus).toarray()
    return tfidf.get_feature_names_out(), X_tfidf

# Provide the correct path to your PDF file
pdf_path = r"C:\Users\Appala nithin\OneDrive\Pictures\Documents\NARESH-IT\AI-PART\24th- Resume discussion\Resume\Resume\Experience\10.pdf"

# Extract and preprocess text from the PDF
raw_text = extract_text_from_pdf(pdf_path)
cleaned_text = preprocess_text(raw_text)

# Check if the cleaned text is empty before applying vectorization
if cleaned_text.strip():
    # If there is text, apply TF-IDF
    corpus = [cleaned_text]
    vocabulary, tfidf_matrix = apply_tfidf(corpus)
    print("Vocabulary:", vocabulary)
    print("TF-IDF Matrix:", tfidf_matrix)
else:
    print("The document is empty or contains only stop words after preprocessing.")
