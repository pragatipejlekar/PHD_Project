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

nltk.download('punkt')

# ============================================================
# 1️⃣ 20-SENTENCE LONG DOCUMENT
# ============================================================

long_document = """
Artificial Intelligence has significantly changed the modern technological landscape.
Organizations across the world are investing heavily in AI research and development.
Machine learning algorithms are being used in healthcare to detect diseases early.
Hospitals rely on predictive analytics to improve patient outcomes.
Financial institutions use AI systems to prevent fraud and manage risks.
Autonomous vehicles depend on deep neural networks for navigation.
Natural language processing enables chatbots and virtual assistants.
Speech recognition systems are improving human computer interaction.
AI driven robotics is transforming manufacturing industries.
Big data plays a crucial role in training intelligent models.
Cloud computing allows scalable AI deployment.
Despite its benefits, AI raises ethical concerns about bias.
Data privacy has become a major challenge.
Governments are introducing regulations to control AI usage.
Researchers are focusing on explainable AI models.
Cybersecurity systems use AI to detect threats in real time.
Educational institutions are adopting AI based learning platforms.
Smart cities integrate AI for traffic and energy management.
Continuous innovation is shaping the future of artificial intelligence.
Responsible development is essential to ensure sustainable growth.
"""

# ============================================================
# 2️⃣ SENTENCE EXTRACTION
# ============================================================

sentences = sent_tokenize(long_document)
print("Total Sentences:", len(sentences))

# ============================================================
# 3️⃣ PREPROCESSING
# ============================================================

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

clean_sentences = [preprocess(s) for s in sentences]

# ============================================================
# 4️⃣ TF-IDF SENTENCE FEATURES
# ============================================================

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_sentences).toarray()

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
    if score > 0.5
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
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

final_summary = abstractive_summary(extractive_summary)

print("\n================ FINAL HYBRID SUMMARY ================\n")
print(final_summary)