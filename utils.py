import pandas as pd
import numpy as np
from rake_nltk import Rake
import yake
from summa import keywords
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')




def text_processing(unvalid_alpha: list, s: str)->str:
    current_str = s
    for x in unvalid_alpha:
        current_str = current_str.replace(x, ' ')
    tokens = current_str.split(' ')
    current_str = ' '.join([token for token in tokens if token != ''])
        
    return current_str

def get_stopwords_list(add_external=True):
    nltk_list = nltk.corpus.stopwords.words('russian') 
    with open('stopwords-ru.txt', 'r', encoding='utf-8') as f:
        external_list = f.read().split('\n')
    final_list = list(set(nltk_list) | set(external_list)) if add_external == True else nltk_list
        
    return final_list

russian_stopwords = get_stopwords_list()


def predict_keyphrases(text, model, n_words=5):
    if model == 'RAKE':
        rake.extract_keywords_from_text(text)
        keyphrases = rake.get_ranked_phrases()
        
    if model == 'YAKE':
        keyphrases = list(map(lambda x: x[0], yake.extract_keywords(text)))
        
    if model == 'TextRank':
        cleaned_text = ' '.join([word.replace('\n', '') for word in text.split(' ') if word.lower() not in russian_stopwords])
        keyphrases = keywords.keywords(text, language = "russian").split("\n")
        
    if model == 'BERT_spacy':
        vectorizer = KeyphraseCountVectorizer(spacy_pipeline="ru_core_news_sm",
                                              stop_words=russian_stopwords,
                                              pos_pattern='<ADJ.*>*<N.*>+')
        bert_keywords = kw_model.extract_keywords(docs=text,
                                                  vectorizer=vectorizer,
                                                  nr_candidates=20,
                                                  top_n=10,
                                                  use_mmr=True,
                                                  diversity=0.3,
                                                 )
        keyphrases = list(map(lambda x: x[0], bert_keywords))
        
    if model == 'BERT_sklearn':
        bert_keywords = kw_model.extract_keywords(docs=text,
                                                  keyphrase_ngram_range=(1,3),
                                                  stop_words=russian_stopwords,
                                                  nr_candidates=20,
                                                  top_n=10,
                                                  use_mmr=True,
                                                  diversity=0.3)
        keyphrases = list(map(lambda x: x[0], bert_keywords))
    
    return keyphrases[:n_words]

def stem_phrase(phrase):
    stemmer = SnowballStemmer("russian")  
    words = phrase.lower().split(' ')
    stemmed_phrase = ' '.join([stemmer.stem(word) for word in words])
    
    return stemmed_phrase

def check_occurence(check_phrase, phrases_list):
    for phrase in phrases_list:
        k = 0
        phrase_granulated = set(phrase.split(' '))
        check_phrase_granulated = check_phrase.split(' ')
        for check_word in check_phrase_granulated:
            if check_word in phrase_granulated:
                k += 1
        if k/len(check_phrase_granulated) > 0.5:
            return True
    
    return False
    
def keywords_precision_score(keyw_true, keyw_pred, k=5):
    if len(keyw_true) == 0 or len(keyw_pred) == 0:
        return 0
        
    n = 0
    keyw_true = [stem_phrase(phrase) for phrase in keyw_true]
    keyw_pred = [stem_phrase(phrase) for phrase in keyw_pred[:k]]
        
    for keyw in keyw_pred:
        if check_occurence(keyw, keyw_true): 
            n+=1
    
    return n/k

def keywords_recall_score(keyw_true, keyw_pred, k=5):
    if len(keyw_true) == 0 or len(keyw_pred) == 0:
        return 0
    
    n_match = 0
    n_true = len(keyw_true)
    keyw_true = [stem_phrase(phrase) for phrase in keyw_true]
    keyw_pred = [stem_phrase(phrase) for phrase in keyw_pred[:k]]
        
    for keyw in keyw_true:
        if check_occurence(keyw, keyw_pred): 
            n_match += 1
    
    return n_match/n_true

def keywords_mean_reciprocal_rank(keyw_true, keyw_pred, k=5):
    if len(keyw_true) == 0 or len(keyw_pred) == 0:
        return 0
    
    n = 0
    keyw_true = [stem_phrase(phrase) for phrase in keyw_true]
    keyw_pred = [stem_phrase(phrase) for phrase in keyw_pred[:k]]
        
    for n_el, keyw in enumerate(keyw_pred):
        if check_occurence(keyw, keyw_true): 
            n = n_el + 1
            break
    
    return 1/n if n != 0 else 0

def keywords_mean_average_precision(keyw_true, keyw_pred, k):
    if len(keyw_true) == 0 or len(keyw_pred) == 0:
        return 0
    
    idxs = []
    keyw_true = [stem_phrase(phrase) for phrase in keyw_true]
    keyw_pred = [stem_phrase(phrase) for phrase in keyw_pred[:k]]
        
    for n_el, keyw in enumerate(keyw_pred):
        if check_occurence(keyw, keyw_true):
            idxs.append(n_el + 1)
            
    if len(idxs) == 0:
        return 0
    
    precisions = [(n_el+1)/ idx for n_el, idx in enumerate(idxs)]
    return sum(precisions)/len(precisions)

def keywords_stem(keyw_true, keyw_pred, k=5):
    n = 0
    keyw_true = [stem_phrase(phrase) for phrase in keyw_true]
    keyw_pred = [stem_phrase(phrase) for phrase in keyw_pred[:k]]
    
    return keyw_true, keyw_pred

def evaluate_extractor(y_true, y_pred, metric, k=5):
    metric_values = [] 
    
    try:
        if metric == 'precision':
            for idx in range(len(y_true)):
                value = keywords_precision_score(y_true[idx], y_pred[idx], k)
                metric_values.append(value)

        if metric == 'recall':
            for idx in range(len(y_true)):
                value = keywords_recall_score(y_true[idx], y_pred[idx], k)
                metric_values.append(value)

        if metric == 'MRR':
            for idx in range(len(y_true)):
                value = keywords_mean_reciprocal_rank(y_true[idx], y_pred[idx], k)
                metric_values.append(value)

        if metric == 'MAP':
            for idx in range(len(y_true)):
                value = keywords_mean_average_precision(y_true[idx], y_pred[idx], k)
                metric_values.append(value)
            
        return np.mean(metric_values)
    
    except KeyError:
        print(y_true[idx])


def extraction_evaluation_report(df, keyw_true_col, models, metrics, size=0.1):
    result = dict()
    df = df.sample(round(size*len(df))).reset_index()
    
    for model_name in models:
        pred_column = 'keyw_' + model_name
        metrics_values = dict()
        for metric_name in metrics:
            metrics_values[metric_name+'_at_5'] = evaluate_extractor(df[keyw_true_col], df[pred_column], metric_name, k=5)
            metrics_values[metric_name+'_at_10'] = evaluate_extractor(df[keyw_true_col], df[pred_column], metric_name, k=10)
        
        result[model_name] = metrics_values
        
    return result