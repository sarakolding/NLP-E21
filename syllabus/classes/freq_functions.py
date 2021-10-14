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
    Takes in a list of spacy Doc objects and return a dictionary of 
    frequencies for each token over all the documents. E.g. {"Aarhus": 20, "the": 2301, ...}
    """
    res_dict = {}
    for doc in docs:
        # create empty list to check whether token appears multiple times in doc
        duplicates = []
        for token in doc:
            if token.text not in duplicates: 
                # if token is not in dict; add token as key and 1 as value
                if token.text not in res_dict:
                    res_dict[token.text] = 1
                # if the token is already in dic; add 1 to the value of that token
                else: 
                    res_dict[token.text] += 1
                duplicates.append(token.text)
    return res_dict

