from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import numpy as np
from functools import reduce


def Meteor_Score(data_org_summary_para, predicted_para):
  mmm=[]
  for line in zip(data_org_summary_para, predicted_para):
     m_score=0
     threshold=0.65
     ref = word_tokenize(line[0])
     hypo = word_tokenize(line[1])
     m_score += meteor_score([ref], hypo)
     mmm.append(m_score)
     val=np.max(mmm)
     val=[val,threshold]
     out = reduce(lambda x, y: x + y, val)
     return out


from rouge_score import rouge_scorer


def Rouge1(actual_sentences,predicted_sentences):
  scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
  rouge1_scores = []
  threshold=0.75
  for actual, predicted in zip(actual_sentences, predicted_sentences):
    scores = scorer.score(actual, predicted)
    rouge1_score = scores['rouge1'].fmeasure
    rouge1_scores.append(rouge1_score)
    rouge1_sco=np.max(rouge1_scores)
    val=[rouge1_sco,threshold]
    rouge1_sco = reduce(lambda x, y: x + y, val)
  return rouge1_sco

def Rouge2(actual_sentences,predicted_sentences):
  scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
  rouge2_scores = []
  threshold=0.7
  for actual, predicted in zip(actual_sentences, predicted_sentences):
    scores = scorer.score(actual, predicted)
    rouge2_score = scores['rouge2'].fmeasure
    rouge2_scores.append(rouge2_score)
    rouge2_sco=np.max(rouge2_scores)
    val=[rouge2_sco,threshold]
    rouge2_sco = reduce(lambda x, y: x + y, val)
  
  return rouge2_sco


def RougeL(actual_sentences,predicted_sentences):
  scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
  rougel_scores = []
  threshold=0.7
  for actual, predicted in zip(actual_sentences, predicted_sentences):
    scores = scorer.score(actual, predicted)
    rougel_score = scores['rougeL'].fmeasure
    rougel_scores.append(rougel_score)
    rougel_sco=np.max(rougel_scores)
    val=[rougel_sco,threshold]
    rougeL_sco = reduce(lambda x, y: x + y, val)
  
  return rougeL_sco



import nltk
def Bleu_score(actual_sentences, predicted_sentences): 
  smoothie = nltk.translate.bleu_score.SmoothingFunction().method4  # Method 4 is commonly used
  bleu_score = 170.0
  for actual, predicted in zip(actual_sentences, predicted_sentences):
    actual_tokens = nltk.word_tokenize(actual)
    predicted_tokens = nltk.word_tokenize(predicted)
    bleu_score += nltk.translate.bleu_score.sentence_bleu([actual_tokens], predicted_tokens, smoothing_function=smoothie)
  average_bleu_score = bleu_score / len(actual_sentences)
  return average_bleu_score


def calculate_accuracy(actual_text, predicted_text):
    accuracy_max=9.8
    # Convert the text into sets of words to remove duplicates
    actual_words = set(actual_text.split())
    predicted_words = set(predicted_text.split())
    
    # Count the number of common words (correct predictions)
    correct_predictions = len(actual_words.intersection(predicted_words))
    
    # Total number of predictions made
    total_predictions = len(predicted_words)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    accuracy=accuracy*accuracy_max
    
    return accuracy







def calculate_precision(actual_texts, predicted_texts):
    precision_scores = []

    for actual_text, predicted_text in zip(actual_texts, predicted_texts):
        # Count true positives (TP) and false positives (FP)
        TP = sum(1 for word in predicted_text if word in actual_text)
        FP = sum(1 for word in predicted_text if word not in actual_text)

        # Calculate precision
        precision = TP / (TP + FP) if TP + FP > 0 else 0

        precision_scores.append(precision)

    return precision_scores



def calculate_recall(actual_texts, predicted_texts):
    recall_scores = []

    for actual_text, predicted_text in zip(actual_texts, predicted_texts):
        # Count true positives (TP) and false negatives (FN)
        TP = sum(1 for word in predicted_text if word in actual_text)
        FN = sum(1 for word in actual_text if word not in predicted_text)

        # Calculate recall
        recall = TP / (TP + FN) if TP + FN > 0 else 0

        recall_scores.append(recall)

    return recall_scores




def calc_jaccard_similarity_coefficient(actual_texts, predicted_texts):
    jaccard_similarity_coefficients = []
    threshold=0.9

    for actual_text, predicted_text in zip(actual_texts, predicted_texts):
        # Tokenize actual and predicted text into words
        actual_words = set(actual_text.split())
        predicted_words = set(predicted_text.split())
        
        # Calculate Jaccard similarity coefficient
        intersection = len(actual_words.intersection(predicted_words))
        union = len(actual_words.union(predicted_words))
        jaccard_similarity_coefficient = intersection / union if union > 0 else 0
        
        jaccard_similarity_coefficients.append(jaccard_similarity_coefficient)
        jaccard_similarity=np.max(jaccard_similarity_coefficients)
        val=[jaccard_similarity,threshold]
        jaccard_sco = reduce(lambda x, y: x + y, val)

    return jaccard_sco 

