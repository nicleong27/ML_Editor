from nltk.tokenize import regexp_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string

import re

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns

from imblearn.over_sampling import SMOTE


def create_word_counts_dict(vectorizer, fit_vectorizer):
    '''
    Word counts dictionary that pairs the word with the number
    of times it occurs throughout the corpus.

    Parameters
    ----------
    text_series: str

    Returns:
    --------
    cleaned_text: str
    '''
    
    bow = vectorizer.get_feature_names()
    
    word_counts = np.asarray(fit_vectorizer.sum(axis=0))[0]
    word_counts_dict = dict(zip(bow, word_counts))
    
    sorted_vocab = sorted(word_counts_dict.items(), key= lambda x: x[1], 
                                                    reverse=True)
    return sorted_vocab

def plot_bar_chart(x, y, title, xlabel, ylabel):
    '''
    Plots bar chart

    Parameters
    ----------
    x : pandas series
    y : pandas series
    title: string
    xlabel: string
    ylabel: string

    Returns:
    --------
    None
    '''
    
    fig, ax = plt.subplots(figsize=(10,7))

    plt.bar(x, y, color='dodgerblue')
    plt.title(title, fontsize=25)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()

def plot_horizontal_bar_chart(x, y, title, xlabel, ylabel):
    '''
    Plots bar chart

    Parameters
    ----------
    x : pandas series
    y : pandas series
    title: string
    xlabel: string
    ylabel: string

    Returns:
    --------
    None
    '''
    
    fig, ax = plt.subplots(figsize=(10,7))

    plt.barh(x, y, color='dodgerblue')
    plt.title(title, fontsize=25)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.invert_yaxis()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()