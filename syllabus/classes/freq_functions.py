from collections import Counter
import os
import spacy
nlp = spacy.load("en_core_web_sm")

def corpus_loader(folder: str): 
    """
    A corpus loader function which takes in a path to a folder and returns a list of strings.
    """
    corpus = []
    for file in os.listdir(folder):
        # current filepath is the folder combined with the name of the current file
        file_path = folder + "/" + str(file)
        # open and read the current file
        with open(file_path, encoding="utf8") as f:
            lines = f.read()
        # add contents of the current file to the corpus
        corpus.append(lines)
    return corpus

def term_freq(tokens) -> dict: 
    """
    Takes in a spacy doc object and returns a dictionary of term frequency for each token.
    """
    # count how many times unique tokens appear in the doc
    counts = Counter([token.text for token in tokens])
    # return a dictionary with token as key and frequency as value
    return {token: count/len(tokens) for (token, count) in counts.items()}

def doc_freqs(docs) -> dict:
    """
    Takes in a list of spacy doc objects and returns a dictionary of frequencies for each token
    over all the documents. E.g. {"Aarhus": 20, "the": 2301, ...}
    """
    dict = {}
    for doc in docs:
        for token in doc:
            # for the first iteration when there are no instances in the dictionary: add all unique
            # tokens in the doc as keys with a value of 1
            if len(dict) == 0:
                dict = {token.text: 1}
            # for the following iterations: if the token is already in the dictionary, add 1 to the count value
            elif token.text in dict:
                dict[token.text] += 1
            # for the following iterations: if the token is not already in the dictionary, add an entrance
            # in the dictionary with the token as key and a value of 1
            elif token.text not in dict:
                dict[token.text] = 1
    return dict
