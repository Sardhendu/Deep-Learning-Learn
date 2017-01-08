# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:11:50 2016

@author: sardendhu
"""
from __future__ import division

import re
import numpy as np
import itertools
import codecs, json
from six.moves import cPickle as pickle

from nltk import sent_tokenize
from nltk import word_tokenize
from gensim import corpora


# from  import CleanText, ProcessTxtGetTerm


content_replacements = [
    {'pattern': '\(Reuters\)', 'repl': ''},
    {'pattern': '\(CNN\)', 'repl': ''},
    {'pattern': '\(AP\)', 'repl': ''},
    {'pattern': '\(Bloomberg\)', 'repl': ''},
    {'pattern': '\(UPI\)', 'repl': ''},
    {'pattern': '/PRNewswire/', 'repl': ''},
    {'pattern': '\(Marketwired -', 'repl': '('},
    {'pattern': '(Gdynia Newsroom)', 'repl': ''},
    {'pattern': 'Source text for Eikon', 'repl':'Source text'},
    {'pattern': 'Further company coverage', 'repl': ''},
    {'pattern': 'Source text', 'repl':''}
    ]

path = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/Word-Nets/articles_json.json'
dictionary_dir = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/Word-Nets/dictionary.txt'
corpora_dir = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/Word-Nets/corpus.pickle'
train_batch_dir = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/Word-Nets/training_batch/'
test_batch_dir = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/Word-Nets/test_batch/'
valid_batch_dir = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/Word-Nets/crossvalid_batch/'



# print (articles.keys())

stopwords = ['A', 'In', 'It', 'I', 'And', 'My', 'He', 'She', 'The', 'When', 'This', 'What', 'Some', 'Many']


class ProcessTxtGetTerm():

    def __init__(self):
        self.prtl_stm_regx = r"'\w*|n'\w*"  # ['\w* - to catch 's of sam's], [n'\w* to catch n't of didn't], [s$ - to catch s of sams], [(?<!s)s$ will capure apples and output apple, but pit fall is that processes becomes processe]
        self.regx_American = re.compile(r'((?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sept|September|Oct|October|Nov|November|Dec|December)[\s](?:[1-9]|[1-2][0-9]|[3][0-1])(?:[\s]|,\s)(?:1[0-9]\d\d|20[0-9][0-9]))') 

        self.regx_British = re.compile(r"((?:[1-9]|[1-2][0-9]|[3][0-1])?[\s]?(?:(?:(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sept|September|Oct|October|Nov|November|Dec|December))?(?:[\s]|,\s)?(?:1[0-9]\d\d|20[0-9][0-9])))")
        self.rgx_ent_partial = re.compile(r'[A-Z\d][A-Za-z\d]*(?:\s+(?:of\s+)?[A-Z\d][A-Za-z\d]*)*', re.DOTALL)
        self.rgx_ent_full = re.compile(r'[A-Z\d][A-Za-z\d]*(?:\s+?[A-Z\d][A-Za-z\d]*)*', re.DOTALL)
        # rgx_ent_full    = re.compile(r'\b[A-Z][\w\'.-]+(?:[\'\s-]\b[A-Z]\w+)*', re.DOTALL)
        '''
            [A-Z\d][A-Za-z\d]* ->  match a word if it starts with a capital or a digit
            (?:                ->  continue matching infinitely as long as
            (?:of )?           ->  the next word is "of" and/or...
            [A-Z\d][A-Za-z\d]* ->  the following word is capitalized
            )*                 ->  continue recursively
        '''   
    def cln_Content(self, content):
        for repl in content_replacements:
            content = re.sub(repl['pattern'], repl['repl'], content, flags=re.IGNORECASE)
        return content

    def get_Dates(self, sent):
        ''' This date function will capture the following (5 May 2011, May 5, 2011, May 5, 5 May, 2011 )'''
        dates = []
        # print 'Extracting all the dates ....................'

        dates_Br = [d.strip() for d in re.findall(self.regx_British, sent)]
        dates_Am = [d.strip() for d in re.findall(self.regx_American, sent)]

        if any(dates_Am):
            return dates_Am
        else:
            return dates_Br

    

    def get_Ent(self, sent):
        sent_arr_nw =[]
        ent_index = np.array([[ent_index.start(0), ent_index.end(0)] for  ent_index in re.finditer(self.rgx_ent_partial, sent)])
        storei1 = 0
        for i in ent_index:
            if i[0]>storei1:
                sent_arr_nw += [wrd for wrd in word_tokenize(sent[storei1:i[0]])]
                sent_arr_nw.append(sent[i[0]:i[1]])
                storei1 = i[1]
        if storei1<len(sent):
            sent_arr_nw += [wrd for wrd in word_tokenize(sent[storei1:len(sent)])]
        return sent_arr_nw


    def get_processed_sent(self, txt):
        # PARSES THE DOCUMENT, CONSTRUCT SENTENCES, EXTRACT TOKENS, REBUILD SENTENCES
        # REURNS A DOC AS SENTENCE ARRAY WITH SEPARATED TOKENS
        ent_trm = []
        date_trm = []
      
        obj_PTGT = ProcessTxtGetTerm()

        txt = obj_PTGT.cln_Content(txt)
        txt = txt.replace('\n', ' ')

        sent_arr = filter(None,[sentence for sentence in sent_tokenize(txt)])
        sent_arr_nw = [obj_PTGT.get_Ent(sent) for sent in sent_arr]
        return np.array(sent_arr_nw)






#==============================================================================
# Get Terms and Bulk Load    
#==============================================================================

class BlkTerm_ExtStrTask():

    def __init__(self):
        self.dictionary = corpora.Dictionary.load_from_text(dictionary_dir)

    def create_dictionary(self, no_of_docs = 10, bulk_doc_no = 5):
        dictionary = self.dictionary
        # query = {"fields":["ids","content","title"], "size" : no_of_docs}
        with open(path, 'r') as f:
            articles = json.load(f)

        print ('BuildCorpus (Bulk)!! The length of ducument from ES is: ', len(articles))
        print ('BuildCorpus (Bulk)!! The length of dictionary is: ', len(dictionary))
        print ('')

        # print articles

        k = 1
        for doc_no, doc in enumerate(articles.values()):   # e196741c-dc63-3534-90ec-e35ec5ff5e17
            try:
                if k > 0 and k <= no_of_docs:#16310
                    print ('BuildCorpus (Bulk)!! Document number is: ', k)
                    # print doc["_id"]

                    sent_arr_nw  =   ProcessTxtGetTerm().get_processed_sent(doc)

                    for arr_nw in sent_arr_nw:
                        # print (arr_nw)
                        dictionary_doc = corpora.Dictionary([arr_nw])
                        dictionary.merge_with(dictionary_doc)  # Merge the new dictionary with the existing
                    # break
                    if (k%bulk_doc_no == 0): # 4000
                        print ('BuildCorpus (Bulk)!! Inside the if condition, the value of k is: ', k)
                        dictionary.save_as_text(dictionary_dir)
                        dictionary = corpora.Dictionary.load_from_text(dictionary_dir)
                        
                    k = k+1
            except KeyError as e:
                print ('Logging Error, The ID %s has no %s'%( e))

        # When all is over insert the last items into dictionary and the corpus
        print ('BuildCorpus (Bulk)!! The outer condition k value is: ', k)
        dictionary.save_as_text(dictionary_dir)
            

        #  We compactify the dictionary (We remove the terms that occurs only once)
        dictionary = corpora.Dictionary.load_from_text(dictionary_dir)
        print ('BuildCorpus (Bulk)!! The length of complete dictionary is : ', len(dictionary))
        ids_occur_once = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq in [1,2,3]]
        print (len(ids_occur_once))
        dictionary.filter_tokens(ids_occur_once)
        print ('BuildCorpus (Bulk)!! The length of terms that occur only ones is: ', len(ids_occur_once))
        dictionary.compactify()
        print ('BuildCorpus (Bulk)!! The length of compactified dictionay is: ', len(dictionary))

        # Fianally We Add the START, END and UNKNOWN Tokens 
        dictionary.merge_with(corpora.Dictionary([["START_TOKEN", "END_TOKEN", "UNK_TOKEN"]]))
        dictionary.save_as_text(dictionary_dir)


    def create_corpus(self, no_of_docs = 10, bulk_doc_no = 2):
        corpus_newdoc = []
        dictionary = self.dictionary
        dictionary_size = len(dictionary)
        with open(path, 'r') as f:
            articles = json.load(f)

        print ('BuildCorpus (Bulk)!! The length of ducument from ES is: ', len(articles))
        print ('BuildCorpus (Bulk)!! The length of dictionary is: ', dictionary_size)
        print ('')
        
        k = 1
        for docID, doc in articles.items():   # e196741c-dc63-3534-90ec-e35ec5ff5e17
            if k > 0 and k <=  no_of_docs:
                print ('BuildCorpus (Bulk)!! Document number is: ', k)

                print ('BuildCorpus (Bulk)!! The docID is: ', docID)

                sent_arr_nw  =   ProcessTxtGetTerm().get_processed_sent(doc)
                for sent_arr in sent_arr_nw:
                    arr = [dictionary.token2id["START_TOKEN"]]
                    # print (sent_arr)
                    for token in sent_arr:
                        try:
                            arr.append(dictionary.token2id[token])
                        except KeyError:
                            # print ('pepepepepepep')
                            arr.append(dictionary.token2id["UNK_TOKEN"])
                    arr += [dictionary.token2id["END_TOKEN"]]
                    # print (arr)
                    corpus_newdoc.append(arr)

                k = k+1
      
        try:
            with open(corpora_dir, 'wb') as f:
                pickle.dump(corpus_newdoc, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', set_filename, ':', e)

        print ('The word corpus can be found at %s ' %corpora_dir)


class CreateBatches():
    def __init__(self):

        self.dictionary = corpora.Dictionary.load_from_text(dictionary_dir)

    def chunks(self, data_in, batch_size):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(data_in), batch_size):
            yield data_in[i:i + batch_size]
        
    # For one batch_element : [1,2,3,4], batch_dataset for that element is [1,2,3] and batch_label is [2,3,4]
    def pad_data_and_labels(self, batch):
        batch_dataset = []
        batch_labels = []
        batch_lenarr = []   # We need the lenght of every row in batch to help RNN (Recurrent network) for learning
        maxlen = len(max(batch,key=len))  # Find the length of the sublist with highest elements
        for array in batch:
            # In the below code we add 1 because we perform padding of 0 to each array in  and our dictionary contains the ID (0), hence we add all the element with 1. so that all the 0 changes to 1 and the zero padding doesnt effect the model we build
            # np.pad(array)
            batch_dataset.append(np.pad(np.add(array[0:len(array)-1], 1), (0,maxlen-len(array)), mode='constant'))
            batch_labels.append(np.pad(np.add(array[1:len(array)], 1), (0,maxlen-len(array)), mode='constant'))
            batch_lenarr.append(len(array)-1)
        return np.vstack(batch_dataset), np.vstack(batch_labels), batch_lenarr

    def create_batches(self, data_in, batch_size):
        for batch in self.chunks(data_in, batch_size):
            batch_dataset, batch_labels, batch_lenarr = self.pad_data_and_labels(batch)
            yield batch_dataset, batch_labels, batch_lenarr

    # THe below function creates the batches for the training, crossvalid and test data
    # Also it creates the labels for each sentence
    def create_train_valid_test_batch(self, prcntg_test_valid, batch_size):
        with open(corpora_dir, 'rb') as f:
            dataset = pickle.load(f)
            if (len(dataset) > 5000):
                np.random.shuffle(dataset)
                test_len = int(np.ceil(len(dataset) * (prcntg_test_valid/100)))
                test_data = dataset[0:test_len]
                valid_data = dataset[test_len:int(np.ceil(test_len+test_len/2))]
                train_data = dataset[int(np.ceil(test_len+test_len/2)):len(dataset)]
                print ('Test data size = ', len(test_data))
                print ('Valid data size = ', len(valid_data))
                print ('Training data size = ', len(train_data))
                np.random.shuffle(train_data)
                
                try:
                    for no, (batch_dataset, batch_labels, batch_lenarr) in enumerate(self.create_batches(train_data, batch_size)):
                        with open(train_batch_dir+'batch'+str(no)+'.pickle', 'wb') as f:
                            batch = {
                                'batch_train_dataset': batch_dataset,
                                'batch_train_labels': batch_labels,
                                'batch_train_lenarr': batch_lenarr
                            }
                            pickle.dump(batch, f, pickle.HIGHEST_PROTOCOL)

                    for no, (batch_dataset, batch_labels, batch_lenarr) in enumerate(self.create_batches(valid_data, batch_size)):
                        with open(valid_batch_dir+'batch'+str(no)+'.pickle', 'wb') as f:
                            batch = {
                                'batch_valid_dataset': batch_dataset,
                                'batch_valid_labels': batch_labels,
                                'batch_valid_lenarr': batch_lenarr
                            }
                            pickle.dump(batch, f, pickle.HIGHEST_PROTOCOL)

                    for no, (batch_dataset, batch_labels, batch_lenarr) in enumerate(self.create_batches(test_data, batch_size)):
                        with open(test_batch_dir+'batch'+str(no)+'.pickle', 'wb') as f:
                            batch = {
                                'batch_test_dataset': batch_dataset,
                                'batch_test_labels': batch_labels,
                                'batch_test_lenarr': batch_lenarr
                            }
                            pickle.dump(batch, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to any one of the train, test or valid dir', e)
                    raise
                    
            else:
                print ("There's not much Data, We Demand Atleast more that 5000")



# BlkTerm_ExtStrTask().create_dictionary(no_of_docs = 2000, bulk_doc_no = 1000)
# BlkTerm_ExtStrTask().create_corpus(no_of_docs = 2000)
CreateBatches().create_train_valid_test_batch(prcntg_test_valid = 10, batch_size = 64)





################################  DUMMY TRY  ################################

# data_in = [[1,2,3,4],[2,3,4,5,6,7], [1,2,3,4,5,6], [4,3,5], [3,4,5,6,7,8,8], [3,3,3,3,4,5,5,6]]
# for no, (a,b,c) in enumerate(CreateBatches().create_batches(data_in, 2)):
#     print (no)
#     print (a)
#     print (b)
#     print (c)
#     print ('')


# with open(train_batch_dir+'batch'+str(1)+'.pickle', 'rb') as f:
#     a = pickle.load(f)
#     print (a)
