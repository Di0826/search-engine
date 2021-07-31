import argparse
import os
import re
from inverted_index import InvertedIndex
from preprocessor import Preprocessor
from similarity_measures import TF_Similarity, TFIDF_Similarity,BM25_Similarity

parser = argparse.ArgumentParser(description='Run all queries on the inverted index.')
parser.add_argument('--new', default=True, help='If True then build a new index from scratch. If False then attempt to'
                                                ' reuse existing index')
parser.add_argument('--sim', default='BM25', help='The type of similarity to use. Should be "TF" or "TFIDF"')
args = parser.parse_args()

index = InvertedIndex(Preprocessor())
index.index_directory(os.path.join('gov', 'documents'), use_stored_index=(not args.new))

sim_name_to_class = {'TF': TF_Similarity,
                     'TFIDF': TFIDF_Similarity, 
                     'BM25': BM25_Similarity}

sim = sim_name_to_class[args.sim]
index.set_similarity(sim)
print(f'Setting similarity to {sim.__name__}')

print()
print('Index ready.')


topics_file = os.path.join('gov', 'topics', 'gov.topics')
runs_file = os.path.join('runs', 'retrieved.runs')

# TODO run queries
"""
You will need to:
    1. Read in the topics_file.
    2. For each line in the topics file create a query string (note each line has both a query_id and query_text,
       you just want to search for the text)  and run this query on index with index.run_query().
    3. Write the results of the query to runs_file IN TREC_EVAL FORMAT
        - Trec eval format requires that each retrieval is on a separate line of the form
          query_id Q0 document_id rank similarity_score MY_IR_SYSTEM
"""
def delete_indexOfquery():
    ''' Delete document id and get queries'''
    LINE_PATTERN =r'\s*\d+\s?(.*)'
    f = open('D:/ANU/Sem2_2020/COMP6490/A1/gov/topics/gov.topics', 'r')
    queries = f.read()
    c = re.compile(LINE_PATTERN)
    lists = []
    lines = queries.split('\n')
    for line in lines:
        r = c.findall(line)
        if r:
            lists.append(r[0])
    return lists
def querydict():
    '''Return a dictionary with document id as key and query as value'''
    queries = delete_indexOfquery()
    query_id = []
    dictquery = {}
    with open('D:/ANU/Sem2_2020/COMP6490/A1/gov/topics/gov.topics', 'r') as query_file:
        for query in query_file:
            query = query.strip().split(' ')
            query_id.append(query[0])
    for i in range(len(query_id)):
        dictquery[query_id[i]]=queries[i]
    return dictquery
def writeToretrieved(queries):
    '''Write the returned results for all of the queries 
    to runs/retrieved.txt in TREC_EVAL format.'''
    fo = open("D:/ANU/Sem2_2020/COMP6490/A1/runs/retrieved.runs", "w+")
    for query_id,query in queries.items():
        doc_sim = index.run_query(query,max_results_returned=10)
        rank = 0
        for doc in doc_sim:
            msg = query_id+" "+"Q0"+" "+ doc[0]+ " " + str(rank) + " " + str(doc[1]) + " "+ "MY_IR_SYSTEM\n"
            fo.write(msg)
            rank += 1
    fo.close
query = querydict()
writeToretrieved = writeToretrieved(query)