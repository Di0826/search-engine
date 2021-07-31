import nltk
from functools import lru_cache
from nltk.corpus import stopwords

class Preprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stermmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.stem = lru_cache(maxsize=10000)(nltk.SnowballStemmer('english').stem)
        #self.list_stopWords=list(set(stopwords.words('english')))
       # self.tokenize = nltk.tokenize.WhitespaceTokenizer().tokenize
        #self.tokenize = nltk.tokenize.RegexpTokenizer().tokenize

    def __call__(self, text):
        #tokens = nltk.WhitespaceTokenizer().tokenize(text)
        #tokens = nltk.regexp_tokenize(text,'\w+|\$[\d\.]+|\S+')
        tokens = nltk.word_tokenize(text)
        tokens = [self.stem(token) for token in tokens]
        return tokens
