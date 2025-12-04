import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Head.Mouth import speak   # make sure speak() is correct

# Make sure nltk assets are downloaded once
nltk.download('punkt')
nltk.download('stopwords')

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        qna_pairs = [line.strip().split(':') for line in lines if ':' in line]
        dataset = [{'question': q, 'answer': a} for q, a in qna_pairs]
        return dataset

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(tokens)

def train_tfidf_vectorizer(dataset):
    corpus = [preprocess_text(qa['question']) for qa in dataset]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def get_answer(question, vectorizer, X, dataset):
    processed_q = preprocess_text(question)
    question_vec = vectorizer.transform([processed_q])
    similarities = cosine_similarity(question_vec, X)
    best_match_index = similarities.argmax()
    return dataset[best_match_index]['answer']

def mind(text):
    dataset_path = r'/home/ak/Desktop/JARVIS/Data/brain_data/qna_dat.txt'
    dataset = load_dataset(dataset_path)
    vectorizer, X = train_tfidf_vectorizer(dataset)
    answer = get_answer(text, vectorizer, X, dataset)
    speak(answer)

if __name__ == "__main__":
    while True:
        user = input("You: ")
        mind(user)
