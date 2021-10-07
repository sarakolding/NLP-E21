from freq_functions import corpus_loader, term_freq, doc_freqs
import spacy
import os
nlp = spacy.load("en_core_web_sm")

# doc = nlp("halli hallo hallo")
# doc1 = nlp("hallo halloej")
# docs2 = [doc, doc1]

# #doc_freqs1 = doc_freqs(docs)
# #print(doc_freqs1)
# print(doc_freqs(docs2))

corpus = corpus_loader("syllabus/classes/data/train_corpus")
docs = [nlp(t) for t in corpus]
#term_freqs = [term_freq(doc) for doc in docs]
#print(term_freqs)

train_freqs = doc_freqs(docs)
print(train_freqs)
