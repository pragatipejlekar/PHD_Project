import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Plus
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from collections import Counter
from nltk.util import ngrams
import math
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from Calculation import Rouge1, Rouge2,RougeL

def perplexity(text):
    words = text.split()
    prob = 1 / len(set(words))
    return pow(prob, -1/len(words))

wnl = WordNetLemmatizer()
from textblob import TextBlob

def build_sparse_graph(features, threshold=0.05):
    similarity = np.dot(features, features.T)
    adjacency = np.zeros_like(similarity)

    for i in range(len(features)):
        for j in range(len(features)):
            if similarity[i][j] > threshold:
                adjacency[i][j] = similarity[i][j]

    # Diagonal expansion
    for i in range(len(features)):
        adjacency[i][i] = similarity[i][i]

    return adjacency

class U_BiTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(U_BiTCN, self).__init__()

        self.branch1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.branch3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)

        self.fc = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(2)

        b1 = F.relu(self.branch1(x))
        b2 = F.relu(self.branch2(x))
        b3 = F.relu(self.branch3(x))

        b1 = torch.mean(b1, dim=2)
        b2 = torch.mean(b2, dim=2)
        b3 = torch.mean(b3, dim=2)

        concat = torch.cat([b1, b2, b3], dim=1)
        return self.fc(concat)
    
def calculate_entropy(features):
    entropy_scores = []

    for vec in features:
        total = np.sum(vec)

        if total == 0:
            entropy_scores.append(0)
            continue

        prob = vec / total
        prob = prob[prob > 0]

        entropy = -np.sum(prob * np.log2(prob))
        entropy_scores.append(entropy)

    return np.array(entropy_scores)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def abstractive_summary(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    summary_ids = bart_model.generate(
        inputs["input_ids"],
        max_length=120,
        min_length=40,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def entropy(text):
    words = word_tokenize(text)
    freq = Counter(words)
    total = sum(freq.values())
    probs = [v/total for v in freq.values()]
    return -sum(p*np.log2(p) for p in probs)
def kl_divergence(p_text, q_text):
    p_words = Counter(word_tokenize(p_text))
    q_words = Counter(word_tokenize(q_text))

    vocab = set(p_words.keys()).union(set(q_words.keys()))
    total_p = sum(p_words.values())
    total_q = sum(q_words.values())

    kl = 0
    for word in vocab:
        p = p_words[word]/total_p if word in p_words else 1e-9
        q = q_words[word]/total_q if word in q_words else 1e-9
        kl += p * math.log(p/q)
    return kl

Ds1 = pd.read_csv("EDGAR-CORPUS-Financial-Summarization.csv")
X1 = Ds1.to_numpy()




Input = X1[:, 0]
Summary = X1[:, 1]



nltk.download("punkt")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


n = 1
Input1 = Input[0:n]
reference_summary = Summary[0:n]

all_metrics = []

for i in range(n):

    

    original = Input1[i]
    reference = reference_summary[i]

    sentences = sent_tokenize(original)

    # ================= TF-IDF =================
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences).toarray()

    # ================= SPARSE GRAPH =================
    adj_matrix = build_sparse_graph(X)
    spatial_features = np.dot(adj_matrix, X)

    # ================= U-BiTCN =================
    model = U_BiTCN(X.shape[1], 64)
    spatial_tensor = torch.FloatTensor(spatial_features)
    temporal_features = model(spatial_tensor).detach().numpy()

    # ================= GSABT =================
    GSABT_features = np.concatenate([spatial_features, temporal_features], axis=1)
    scaler = MinMaxScaler()
    GSABT_features = scaler.fit_transform(GSABT_features)

    entropy_scores = calculate_entropy(GSABT_features)

    selected_sentences = [
        sentences[j] for j, score in enumerate(entropy_scores)
        if score > np.mean(entropy_scores)
    ]

    extractive_summary = " ".join(selected_sentences)

    final_summary = abstractive_summary(extractive_summary)
    final_summary =extractive_summary
    # ==========================================================
    # 🔥 20 PERFORMANCE METRICS (INSIDE LOOP)
    # ==========================================================

    orig_words = word_tokenize(original)
    sum_words = word_tokenize(final_summary)

    orig_sent = sent_tokenize(original)
    sum_sent = sent_tokenize(final_summary)

    jaccard = len(sum_words) / len(orig_words)
    sentence_reduction = len(sum_sent) / len(orig_sent)
    stop_word_ratio = 1 - jaccard
    vocab_richness = len(set(sum_words))
    type_token_ratio = len(set(sum_words)) / len(sum_words)
    avg_sentence_length = np.mean([len(word_tokenize(s)) for s in sum_sent])
    compression_ratio = len([w for w in sum_words if w.lower() in stop_words]) / len(sum_words)
    lexical_density = 1 - compression_ratio

    
    tfidf = TfidfVectorizer().fit_transform([reference, final_summary])
   
    

    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, final_summary)




    # Semantic
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode([reference, final_summary])
    semantic_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Entropy
    entropy_original = entropy(original)
    entropy_summary = entropy(final_summary)
    entropy_difference = abs(entropy_original - entropy_summary)

    kl_value = kl_divergence(original, final_summary)
    coverage = len(set(sum_words) & set(orig_words)) / len(set(orig_words))

    bigram_list = list(ngrams(sum_words, 2))
    redundancy = 1 - (len(set(bigram_list)) / len(bigram_list)) if len(bigram_list)>0 else 0
    R1=Rouge1(reference, final_summary)
    R2=Rouge2(reference, final_summary)
    Rl=RougeL(reference, final_summary)   
   

    P, R, F1 = bert_score([final_summary], [reference], lang="en")
    bert_f1 = F1.mean().item()

    bleu = sentence_bleu([reference.split()], final_summary.split())
    meteor = meteor_score([reference.split()], final_summary.split())

    orig_set = set(reference.split())
    sum_set = set(final_summary.split())
    Semantic_Similarity = len(sum_set - orig_set) / len(sum_set) if len(sum_set)>0 else 0

    ppl = perplexity(final_summary)
    length_stability = abs(len(reference.split()) - len(final_summary.split()))

    ref_bigrams = set(ngrams(reference.split(), 2))
    cand_bigrams = set(ngrams(final_summary.split(), 2))
    bigram_overlap = len(ref_bigrams & cand_bigrams) / len(ref_bigrams) if len(ref_bigrams)>0 else 0
    

















