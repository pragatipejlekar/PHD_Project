from rouge_score import rouge_scorer
from functools import reduce
from functools import reduce
import numpy as np

def Rouge1(actual_sentences,predicted_sentences):
  scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
  rouge1_scores = []
  threshold=0.17
  for actual, predicted in zip(actual_sentences, predicted_sentences):
    scores = scorer.score(actual, predicted)
    rouge1_score = scores['rouge1'].fmeasure
    rouge1_scores.append(rouge1_score)
    rouge1_sco=np.max(rouge1_scores)
    val=[rouge1_sco,threshold]
    rouge1_sco = reduce(lambda x, y: x - y, val)
  return rouge1_sco


def Rouge2(actual_sentences,predicted_sentences):
  scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
  rouge2_scores = []
  threshold=0.8
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
  threshold=0.18
  for actual, predicted in zip(actual_sentences, predicted_sentences):
    scores = scorer.score(actual, predicted)
    rougel_score = scores['rougeL'].fmeasure
    rougel_scores.append(rougel_score)
    rougel_sco=np.max(rougel_scores)
    val=[rougel_sco,threshold]
    rougeL_sco = reduce(lambda x, y: x - y, val)
  
  return rougeL_sco
