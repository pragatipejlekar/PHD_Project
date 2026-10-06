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
import pyarrow.parquet as pa

wnl = WordNetLemmatizer()
from textblob import TextBlob



table = pa.read_table('Dataset2\\test-00000-of-00001.parquet')
df = table.to_pandas()

X1=df.to_numpy()

Input = X1[:, 0]
Summary = X1[:, 1]

n=20


Input1=Input[0:n]
# reference_summary=Summary[0:n]


nltk.download("punkt")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
documents=[]

org_text=[]
splitted_out=[]
normalized_out=[]
lowercase_out=[]
stopword_out=[]
lemmatize_out=[]
actual_text=[]
Input2 = Input1.tolist()

# actal_text= reference_summary.tolist()
for i in range(0,1):#len(combined_list)
    
  input_str=Input2[i]
  org_text.append(input_str)
  
  
  
  sentences = sent_tokenize(input_str)
  
  
 
  
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences).toarray()



num_sentences, feature_dim = X.shape
print("TF-IDF Shape:", X.shape)

# ============================================================
# 5️⃣ GRAPH SPARSE ATTENTION (Spatial)
# Diagonal expansion + zero filling
# ============================================================

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

adj_matrix = build_sparse_graph(X)

# Spatial features
spatial_features = np.dot(adj_matrix, X)
print("Spatial Feature Shape:", spatial_features.shape)

# ============================================================
# 6️⃣ U-BiTCN (Temporal Features)
# ============================================================

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

model = U_BiTCN(feature_dim, 64)

spatial_tensor = torch.FloatTensor(spatial_features)
temporal_features = model(spatial_tensor).detach().numpy()

print("Temporal Feature Shape:", temporal_features.shape)

# ============================================================
# 7️⃣ COMBINE GSABT FEATURES
# ============================================================

GSABT_features = np.concatenate([spatial_features, temporal_features], axis=1)

# Normalize to avoid entropy instability
scaler = MinMaxScaler()
GSABT_features = scaler.fit_transform(GSABT_features)

print("Final GSABT Shape:", GSABT_features.shape)

# ============================================================
# 8️⃣ STABLE ENTROPY CALCULATION
# ============================================================

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

entropy_scores = calculate_entropy(GSABT_features)

print("\nEntropy Scores:")
print(entropy_scores)

# ============================================================
# 9️⃣ SENTENCE SELECTION (Threshold = 0.5)
# ============================================================

selected_sentences = [
    sentences[i] for i, score in enumerate(entropy_scores)
    if score > 9
]

extractive_summary = " ".join(selected_sentences)

print("\n================ EXTRACTIVE SUMMARY ================\n")
print(extractive_summary)

# ============================================================
# 🔟 ABSTRATIVE SUMMARIZATION (BART)
# ============================================================

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

final_summary = abstractive_summary(extractive_summary)

print("\n================ FINAL HYBRID SUMMARY ================\n")
print(final_summary)
  
  
  
# ============================================================
# 11️⃣ 20 PERFORMANCE METRICS
# ============================================================

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from collections import Counter
from nltk.util import ngrams
import math

original = org_text[0]
summary = extractive_summary

# -------------------------------
# BASIC COUNTS
# -------------------------------
orig_words = word_tokenize(original)
sum_words = word_tokenize(summary)

orig_sent = sent_tokenize(original)
sum_sent = sent_tokenize(summary)

# 1 Compression Ratio
compression_ratio = len(sum_words) / len(orig_words)

# 2 Sentence Reduction Ratio
sentence_reduction = len(sum_sent) / len(orig_sent)

# 3 Word Reduction Ratio
word_reduction = 1 - compression_ratio

# 4 Vocabulary Richness
vocab_richness = len(set(sum_words))

# 5 Type-Token Ratio
type_token_ratio = len(set(sum_words)) / len(sum_words)

# 6 Avg Sentence Length
avg_sentence_length = np.mean([len(word_tokenize(s)) for s in sum_sent])

# 7 Stopword Ratio
stop_ratio = len([w for w in sum_words if w.lower() in stop_words]) / len(sum_words)

# 8 Lexical Density
lexical_density = 1 - stop_ratio

# -------------------------------
# SIMILARITY METRICS
# -------------------------------

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform([original, summary])
cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

# 10 Jaccard Similarity
jaccard = len(set(orig_words) & set(sum_words)) / len(set(orig_words) | set(sum_words))

# 11-13 ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
rouge_scores = scorer.score(original, summary)

rouge1 = rouge_scores['rouge1'].fmeasure
rouge2 = rouge_scores['rouge2'].fmeasure
rougeL = rouge_scores['rougeL'].fmeasure

# -------------------------------
# SEMANTIC METRICS
# -------------------------------

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([original, summary])
semantic_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# -------------------------------
# INFORMATION METRICS
# -------------------------------

def entropy(text):
    words = word_tokenize(text)
    freq = Counter(words)
    total = sum(freq.values())
    probs = [v/total for v in freq.values()]
    return -sum(p*np.log2(p) for p in probs)

entropy_original = entropy(original)
entropy_summary = entropy(summary)
entropy_difference = abs(entropy_original - entropy_summary)

# 18 KL Divergence
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

kl_value = kl_divergence(original, summary)

# 19 Coverage Score
coverage = len(set(sum_words) & set(orig_words)) / len(set(orig_words))

# 20 Redundancy Score
bigram_list = list(ngrams(sum_words, 2))
redundancy = 1 - (len(set(bigram_list)) / len(bigram_list)) if len(bigram_list)>0 else 0

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n================ 20 PERFORMANCE METRICS ================\n")

metrics = {
"Compression Ratio": compression_ratio,
"Sentence Reduction Ratio": sentence_reduction,
"Word Reduction Ratio": word_reduction,
"Vocabulary Richness": vocab_richness,
"Type Token Ratio": type_token_ratio,
"Average Sentence Length": avg_sentence_length,
"Stopword Ratio": stop_ratio,
"Lexical Density": lexical_density,
"Cosine Similarity": cos_sim,
"Jaccard Similarity": jaccard,
"ROUGE-1": rouge1,
"ROUGE-2": rouge2,
"ROUGE-L": rougeL,
"Semantic Similarity": semantic_sim,
"Entropy Original": entropy_original,
"Entropy Summary": entropy_summary,
"Entropy Difference": entropy_difference,
"KL Divergence": kl_value,
"Coverage Score": coverage,
"Redundancy Score": redundancy
}

for k,v in metrics.items():
    print(f"{k}: {v}") 
    
    
# ============================================================
# ADDITIONAL 8 ADVANCED METRICS
# ============================================================

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score


reference = original
candidate = summary

# 1️⃣ BERTScore
P, R, F1 = bert_score([candidate], [reference], lang="en")
bert_f1 = F1.mean().item()

# 2️⃣ BLEU
bleu = sentence_bleu([reference.split()], candidate.split())

# 3️⃣ METEOR
meteor = meteor_score([reference.split()], candidate.split())

# 4️⃣ Novelty Ratio
orig_set = set(reference.split())
sum_set = set(candidate.split())
novelty = len(sum_set - orig_set) / len(sum_set)



# 6️⃣ Perplexity (Simple Approximation)
def perplexity(text):
    words = text.split()
    prob = 1 / len(set(words))
    return pow(prob, -1/len(words))

ppl = perplexity(candidate)

# 7️⃣ Length Ratio Stability
length_stability = abs(len(reference.split()) - len(candidate.split()))

# 8️⃣ Bigram Overlap Ratio
from nltk.util import ngrams
ref_bigrams = set(ngrams(reference.split(), 2))
cand_bigrams = set(ngrams(candidate.split(), 2))
bigram_overlap = len(ref_bigrams & cand_bigrams) / len(ref_bigrams) if len(ref_bigrams)>0 else 0

# ============================================================
print("\n============= ADDITIONAL 8 METRICS =============\n")

extra_metrics = {
"BERTScore F1": bert_f1,
"BLEU Score": bleu,
"METEOR Score": meteor,
"Novelty Ratio": novelty,

"Perplexity": ppl,
"Length Stability Difference": length_stability,
"Bigram Overlap Ratio": bigram_overlap
}

for k,v in extra_metrics.items():
    print(f"{k}: {v}")
  
  
# ============================================================
# 🔥 MORE ADVANCED SUMMARIZATION METRICS (Research Grade)
# ============================================================

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from sklearn.metrics import mutual_info_score

# -------------------------------
# 1️⃣ Distinct-1
# -------------------------------
distinct1 = len(set(candidate.split())) / len(candidate.split())

# -------------------------------
# 2️⃣ Distinct-2
# -------------------------------
candidate_bigrams = list(ngrams(candidate.split(), 2))
distinct2 = len(set(candidate_bigrams)) / len(candidate_bigrams) if len(candidate_bigrams)>0 else 0

# -------------------------------
# 3️⃣ Self-BLEU (Redundancy Measure)
# -------------------------------
smooth = SmoothingFunction().method1
self_bleu = sentence_bleu([candidate.split()], candidate.split(), smoothing_function=smooth)

# -------------------------------
# 4️⃣ Jensen-Shannon Divergence
# -------------------------------
def get_distribution(text):
    words = word_tokenize(text)
    freq = Counter(words)
    total = sum(freq.values())
    return np.array([freq[w]/total for w in sorted(freq.keys())])

orig_dist = get_distribution(original)
sum_dist = get_distribution(summary)

# Align lengths
min_len = min(len(orig_dist), len(sum_dist))
js_div = jensenshannon(orig_dist[:min_len], sum_dist[:min_len])

# -------------------------------
# 5️⃣ Mutual Information
# -------------------------------
orig_tokens = word_tokenize(original)
sum_tokens = word_tokenize(summary)

mi_score = mutual_info_score(orig_tokens[:min_len], sum_tokens[:min_len])

# -------------------------------
# 6️⃣ Topic Similarity (LDA-based)
# -------------------------------
vectorizer = CountVectorizer(stop_words='english')
doc_term_matrix = vectorizer.fit_transform([original, summary])

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(doc_term_matrix)

topic_dist = lda.transform(doc_term_matrix)
topic_similarity = cosine_similarity([topic_dist[0]], [topic_dist[1]])[0][0]



# ============================================================
print("\n============= 🔥 ADVANCED RESEARCH METRICS =============\n")

advanced_metrics = {
"Distinct-1": distinct1,
"Distinct-2": distinct2,
"Self-BLEU": self_bleu,
"Jensen-Shannon Divergence": js_div,
"Mutual Information": mi_score,
"Topic Similarity": topic_similarity,

}

for k,v in advanced_metrics.items():
    print(f"{k}: {v}")
  
  
  
# ============================================================
# 🔥 ERROR-BASED & CLASSIFICATION STYLE METRICS
# ============================================================

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import Levenshtein
from jiwer import wer, cer

reference = original
candidate = summary

# -------------------------------
# 1️⃣ Word Error Rate (WER)
# -------------------------------
wer_score = wer(reference, candidate)

# -------------------------------
# 2️⃣ Character Error Rate (CER)
# -------------------------------
cer_score = cer(reference, candidate)

# -------------------------------
# 3️⃣ Normalized Levenshtein Similarity
# -------------------------------
lev_distance = Levenshtein.distance(reference, candidate)
max_len = max(len(reference), len(candidate))
norm_lev_similarity = 1 - (lev_distance / max_len)

# -------------------------------
# 4️⃣ MAPE (Word Count Based)
# -------------------------------
orig_len = len(reference.split())
sum_len = len(candidate.split())

if orig_len != 0:
    mape = abs(orig_len - sum_len) / orig_len
else:
    mape = 0

# -------------------------------
# 5️⃣ Token-Level Accuracy, Precision, Recall, F1
# -------------------------------

ref_tokens = reference.split()
cand_tokens = candidate.split()

# Create binary presence vectors
all_tokens = list(set(ref_tokens + cand_tokens))

ref_vector = [1 if token in ref_tokens else 0 for token in all_tokens]
cand_vector = [1 if token in cand_tokens else 0 for token in all_tokens]

accuracy = accuracy_score(ref_vector, cand_vector)
precision = precision_score(ref_vector, cand_vector)
recall = recall_score(ref_vector, cand_vector)
f1 = f1_score(ref_vector, cand_vector)

# ============================================================
print("\n============= ERROR & CLASSIFICATION METRICS =============\n")

error_metrics = {
"Word Error Rate (WER)": wer_score,
"Character Error Rate (CER)": cer_score,
"Normalized Levenshtein Similarity": norm_lev_similarity,
"MAPE": mape,
"Accuracy": accuracy,
"Precision": precision,
"Recall": recall,
"F1-Score": f1
}

for k,v in error_metrics.items():
    print(f"{k}: {v}")  
  
  
  
  
  
  
  
  
  
  
  
  
  
  