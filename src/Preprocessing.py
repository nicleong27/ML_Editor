from nltk.tokenize import regexp_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
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
    
    stemmer = WordNetLemmatizer()
    punctuations = set(string.punctuation)
    
    # remove http links at end of tweets
    cleaned_text = text_series.apply(lambda row: re.sub(r'http\S+', '', row))
    
    # remove emojis from tweet
    cleaned_text = cleaned_text.apply(lambda row: 
                                    re.sub(emoji.get_emoji_regexp(), r'', row))
    
    # remove punctuations
    cleaned_text = cleaned_text.apply(lambda row: re.sub(r'[^\w\s]', '', row))
    
    # remove standalone numbers from tweets, tweets embedded in words will not 
    # be removed
    cleaned_text = cleaned_text.apply(lambda row: re.sub(r'\b\d+\b', '', row))
    
    cleaned_text = cleaned_text.apply(lambda row: row.strip())
    
    # tokenize words in order to lemmatize
    cleaned_text = cleaned_text.apply(lambda row: word_tokenize(row.lower()))
    cleaned_text = cleaned_text.apply(lambda row: 
                        ' '.join([stemmer.lemmatize(word) for word in row]))
    return cleaned_text