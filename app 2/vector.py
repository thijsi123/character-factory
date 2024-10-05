import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Download necessary NLTK data
nltk.download('punkt', quiet=True)


class AdvancedVectorStorage:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectors = []
        self.texts = []

    def add_text(self, text):
        chunks = self.chunk_text(text)
        for chunk in chunks:
            self.texts.append(chunk)
            self.vectors.append(self.model.encode(chunk))

    def chunk_text(self, text, max_length=200):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            if current_length + len(word) > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def query(self, query_text, top_k=5):
        query_vector = self.model.encode(query_text)
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(self.texts[i], similarities[i]) for i in top_indices]


def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    main_content = soup.find('div', class_='mw-parser-output')

    if main_content:
        text_content = []
        for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text_content.append(element.get_text())
        return ' '.join(text_content)
    else:
        return soup.get_text()


def answer_question_with_llm(full_context, embedding_context, question):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "jinaai/reader-lm-0.5b"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    prompt = f"""Please answer the following question based solely on the given context. If the answer is not in the context, say "I don't have enough information to answer this question."

Context: {embedding_context}

Question: {question}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1
    )

    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def answer_question(storage, full_context, question):
    results = storage.query(question, top_k=5)
    embedding_context = " ".join([text for text, _ in results])

    print("Relevant context from embedding model:")
    for text, similarity in results:
        print(f"- {text} (Similarity: {similarity:.2f})")

    llm_answer = answer_question_with_llm(full_context, embedding_context, question)

    return llm_answer


# Example usage
storage = AdvancedVectorStorage()

url = 'https://zelda.fandom.com/wiki/Link'
content = scrape_website(url)

print(f"Content length: {len(content)}")

storage.add_text(content)

question = "Who is Link? What is Link personality? What is link summary? Link gender? What does Link look like? What is Link appearance?"
answer = answer_question(storage, content, question)

print(f"\nQuestion: {question}")
print(f"LLM Answer: {answer}")