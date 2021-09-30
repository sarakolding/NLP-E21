import spacy
import os 
from collections import Counter

## WHAT IS THE USE/BENEFIT OF A DOC OBJECT?
# container for accessing linguistic annotations
# includes information about the tokens (e.g. whitespace)

## TWO WAYS TO CREATE A DOC OBJECT
# Construction 1
# nlp = spacy.load("en_core_web_sm")
# doc = nlp("Some text")

# Construction 2
# words = ["hello", "world", "!"]
# spaces = [True, False, False]
# doc = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces) # classes are denoted by uppercase


## CORPUS LOADER
def corpus_loader(folder: str): #changed from List
    """
    A corpus loader function which takes in a path to a 
    folder and returns a list of strings.
    """

    corpus = []
    for file in os.listdir(folder):
        file_path = folder + "/" + str(file)
        with open(file_path, encoding="utf8") as f:
            lines = f.read()
        corpus.append(lines)
    return corpus

corpus = corpus_loader("/work/NLP-E21/syllabus/classes/data/train_corpus")

## EXERCISE II
nlp = spacy.load("en_core_web_sm")

docs = [nlp(t) for t in corpus]

def nlp_filter(doc):
    lemmas = []
    for token in doc: 
        if token.pos_ in ['NOUN', 'ADJ', 'VERB']:
            lemmas.append(token.lemma_)
    return lemmas

all_lemmas = [nlp_filter(doc) for doc in docs]

## EXERCISE III
pos_counts = Counter([token.pos_ for token in docs[0]])
[(pos, count/len(docs[0])) for pos, count in pos_counts.items()]

## EXERCISE IV
#token: 'reporter'
#token.head: 'admitted'
#take indices from each and subtract

def calculate_mdd(text):
    t_sum = 0
    t_size = 0
    doc = nlp(text)

    for token in doc:
        dist = abs(token.i - token.head.i)
        t_sum += dist
        if dist != 0:
            t_size += 1
        # print(token, token.head, dist)

    mdd = t_sum / t_size
    return mdd


print(calculate_mdd("The reporter who attacked the senator admitted the error."))