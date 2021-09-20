import spacy
import os 

nlp = spacy.load("en_core_web_sm")

## WHAT IS THE USE/BENEFIT OF A DOC OBJECT?
# container for accessing linguistic annotations

## TWO WAYS TO CREATE A DOC OBJECT
# Construction 1
#doc = nlp("Some text")

# Construction 2
#words = ["hello", "world", "!"]
#spaces = [True, False, False]
#doc = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)


## CORPUS LOADER
def corpus_loader(folder: str) -> list[str]: #changed from List
    """
    A corpus loader function which takes in a path to a 
    folder and returns a list of strings.
    """

    corpus = []
    for file in os.listdir(folder):
        file_path = folder + "/" + str(file)
        with open(file_path, encoding="utf8") as f:
            lines = f.read()
        corpus.append([lines])

corpus1 = corpus_loader("C:/Users/sarak/OneDrive - Aarhus universitet/COGNITIVE SCIENCE/SEVENTH SEMESTER/NATURAL LANGUAGE PROCESSING/NLP-E21/syllabus/classes/data/train_corpus")
print(corpus1[1])
