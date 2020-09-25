import spacy
from nltk.stem.wordnet import WordNetLemmatizer
import string

import re

import emoji


def extract_emojis(string):
    '''
    Extracts any emojis from string

    Parameters
    ----------
    string: str

    Returns:
    --------
    String of emojis found in text
    '''
    
    return ''.join(c for c in string if c in emoji.UNICODE_EMOJI)

def clean_text(text_series):
    '''
    Cleans text by removing http links, emojis, standlone numbers,
    and removes any beginning or trailing whitespaces. 

    Parameters
    ----------
    text_series: str

    Returns:
    --------
    cleaned_text: str
    '''
    
    nlp = spacy.load('en_core_web_sm')
    lemm = WordNetLemmatizer()
    
    # remove http links at end of tweets
    cleaned_text = text_series.apply(lambda row: re.sub(r'http\S+', '', row))
    
    # remove emojis from tweet
    cleaned_text = cleaned_text.apply(lambda row: 
                                    re.sub(emoji.get_emoji_regexp(), r'', row))
    
    # remove punctuations, standalone numbers (numbers embedded in words will
    # not be removed), lemmatize words
    cleaned_text = cleaned_text.apply(lambda row: nlp(row.lower()))
    cleaned_text = cleaned_text.apply(lambda row: 
                        ' '.join(lemm.lemmatize(word.orth_) for word in row 
                        if not word.is_punct | (word.pos_ in ['NUM'])))
    
    
    return cleaned_text