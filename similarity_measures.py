from abc import abstractmethod
from collections import defaultdict
from math import log, sqrt

def get_zero():
    return 0


def get_empty_postings():
    return defaultdict(get_zero)

class CosineSimilarity:
    """
    This class calculates a similarity score between a given query and all documents in an inverted index.
    """
    
    def __init__(self, postings):
        self.postings = postings
        self.doc_to_norm = dict()
        self.set_document_norms()

    def __call__(self, query):
        doc_to_score = defaultdict(lambda: 0)
        self.get_scores(doc_to_score, query)
        return doc_to_score
    

    @abstractmethod
    def set_document_norms(self):
        """
        Set self.doc_to_norm to contain the norms of every document.
        """
        pass

    @abstractmethod
    def get_scores(self, doc_to_score, query):
        """
        For each document add an entry to doc_to_score with this document's similarity to query.
        """
        pass


class TF_Similarity(CosineSimilarity):
    def set_document_norms(self):
        for doc, token_counts in self.postings.doc_to_token_counts.items():
            self.doc_to_norm[doc] = sqrt(sum([tf ** 2 for tf in token_counts.values()]))

    def get_scores(self, doc_to_score, query):
        for token, query_term_frequency in query.items():
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                doc_to_score[doc] += query_term_frequency * document_term_frequency / self.doc_to_norm[doc]


class TFIDF_Similarity(CosineSimilarity):
    # TODO implement the set_document_norms and get_scores methods.
    # Get rid of the NotImplementedErrors when you are done.
    def IDF(self):
        '''Return a dictionary with words as keys and idf as values'''
        tokens = defaultdict(lambda: 0)
        n = len(self.postings.doc_to_token_counts.keys())
        for token in self.postings.token_to_doc_counts.keys():
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                if document_term_frequency>0:
                    tokens[token]+=1
        for token,fre in tokens.items():
            tokens[token] = log(n/fre,10)
        return tokens   
    def TFIDF(self):
        '''Return a dictionary with documents and words as keys and tf-idf as values'''
        idf = self.IDF()
        tfidf = defaultdict(get_empty_postings)
        for doc, token_counts in self.postings.doc_to_token_counts.items():
            for token in token_counts:
                tfidf[doc][token] = idf[token]*token_counts[token]
        return tfidf 
        
    def set_document_norms(self):
        tfidf = self.TFIDF()
        for doc, idf in tfidf.items():
            self.doc_to_norm[doc] = sqrt(sum([tf ** 2 for tf in idf.values()]))

    def get_scores(self, doc_to_score, query):
        queryTFIDF = defaultdict(lambda: 0)
        idf = self.IDF()
        tfidf = self.TFIDF()
        for token, query_term_frequency in query.items():   
            queryTFIDF[token] = query_term_frequency*idf[token]
        for token, query_term_frequency in query.items():
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                doc_to_score[doc] += queryTFIDF[token] * tfidf[doc][token] / self.doc_to_norm[doc]
                
class BM25_Similarity(CosineSimilarity):
    
    #def set_document_norms(self):       
    def get_scores(self, doc_to_score, query):
        k1 = 2.0
        b = 0.75
        s = 0
        num_doc = 0
        for doc, token_counts in self.postings.doc_to_token_counts.items():
            s += len(token_counts.keys())
            num_doc+=1
        avgdl =  s/num_doc     
        df = defaultdict(lambda: 0)
        idf = defaultdict(lambda: 0)
        for token, query_term_frequency in query.items():   
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                if document_term_frequency>0:
                    df[token]+=1
        for token, num in df.items():
            idf[token] = log((num_doc-num+0.5)/(num+0.5),10)
        for token, query_term_frequency in query.items():
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                 doc_to_score[doc] += (idf[token]*document_term_frequency*(k1+1))/(document_term_frequency+k1*(1-b+b*len(self.postings.doc_to_token_counts[doc])/avgdl))
            
        
