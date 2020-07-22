import re

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, GridSearchCV

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns

def get_conf_matrix(y_test, y_pred):
    '''
    Creates confusion matrix
    
    Parameters
    ----------
    y_test: array
    y_pred: array
    
    Returns:
    --------
    None
    '''
    
    cm = confusion_matrix(y_test, y_pred)

    # flip confusion matrix, so that confusion matrix is properly ordered
    cm_flip = np.flip(cm, 0)
    cm_flip2 = np.flip(cm_flip, 1)
    df_cm = pd.DataFrame(cm_flip2, index=('Correct', 'Incorrect'), 
                                    columns=('Correct', 'Incorrect'))

    fig, ax = plt.subplots(figsize=(10,7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

def print_acc_prec_recall(y_test, y_pred):
    '''
    Prints accuracy, precision, and recall scores
    
    Parameters
    ----------
    y_test: array
    y_pred: array
    
    Returns:
    --------
    None
    '''
    
    print('Accuracy score: ', accuracy_score(y_test, y_pred))
    print('Precision: ', precision_score(y_test, y_pred))
    print('Recall: ', recall_score(y_test, y_pred))

def plot_roc_curve(y_hat, y_test, classifier):
    '''
    Plots ROC curve
    
    Parameters
    ----------
    y_hat: array
    y_test: array
    classifier: str
                name of classifier
    
    Returns:
    --------
    None
    '''
    
    # retrieve possibilities of positive class
    y_hat = y_hat[:, 1]
    # plot no skill ROC curve
    fig, ax = plt.subplots(figsize=(10,7))
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # calculate fpr, tpr
    fpr, tpr, _ = roc_curve(y_test, y_hat)
    roc_auc = auc(fpr, tpr)
    # plot precision ROC curve
    
    plt.plot(fpr, tpr, marker='.', label=classifier + ', AUC: %0.2f' % roc_auc, 
                                    color='dodgerblue')
    
    plt.title('ROC Curve', fontsize=25)
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.tight_layout()

def plot_prec_recall(y_hat, y_test):
    '''
    Plots precision-recall curve
    
    Parameters
    ----------
    y_hat: array
    y_test: array
    
    Returns:
    --------
    None
    '''
    
    # retrieve possibilities of positive class
    y_hat = y_hat[:, 1]
    
    # plot no skill ROC curve
    fig, ax = plt.subplots(figsize=(10,7))
 
    # calculate precision and recall
    precision, recall, _ = precision_recall_curve(y_test, y_hat)
    # plot precision recall curve
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No skill')
    plt.plot(recall, precision, marker='.', label='Precision/Recall', 
                                            color='dodgerblue')
    
    plt.title('Precision/Recall Curve', fontsize=25)
    ax.set_xlabel('Recall', fontsize=20)
    ax.set_ylabel('Precision', fontsize=20)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.tight_layout()

def tweak_tresholds(thresholds, X_test, y_test, model):
    '''
    Prints and returns accuracy, precision, and recall score for different
    thresholds. 
    
    Parameters
    ----------
    thresholds: list
                range of thresholds
    X_test: sparse matrix
    y_test: array
    model: model instance
    
    Returns:
    --------
    thresholds: list
    precision_scores: list
    recall_scores: list
    '''

    precision_scores = []
    recall_scores = []
    
    for threshold in thresholds:
        y_pred = (model.predict_proba(X_test)[:, 1] > threshold).astype(int)
        print('Threshold of {}'.format(threshold))
        print_acc_prec_recall(y_test, y_pred)
        print('\n')
        
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test,y_pred))
        
    return thresholds, precision_scores, recall_scores