from nltk.tokenize import regexp_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import re

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style('darkgrid')

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
    word_counts_dict = {}

    for word, count in sorted_vocab:
        if word not in word_counts_dict:
            word_counts_dict[word] = 0
        word_counts_dict[word] = count
    
    return word_counts_dict

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

def random_color_func(word=None, font_size=None, position=None, orientation=None,
                        font_path=None, random_state=None):
    '''
    Randomly generates colors for WorldCloud
    '''
    h = int(360 * 140 / 255)
    s = int(100 * 255 /255)
    l = int(100 * float(random_state.randint(40, 120)) / 255)

    return 'hsl({}, {}%, {}%)'.format(h, s, l)

def create_wordcloud(text):
    '''
    Creates wordcloud based on dataframe column name

    Parameters
    ----------
    df: pandas dataframe
    col_name : str 
        Text column name looking to plot

    Returns:
    --------
    None
    '''
    wordcloud = WordCloud(background_color='white', 
                color_func=random_color_func).generate_from_frequencies(text)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')