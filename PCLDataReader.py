##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

from abc import ABC, abstractmethod
from itertools import islice
from collections import Counter
from html import unescape
from lxml import etree
from sklearn.feature_extraction import DictVectorizer
import sys

#####################################################################
# HELPER FUNCTIONS
#####################################################################


def do_xml_parse(fp, tag, max_elements=None, progress_message=None):
    """ 
    Parses cleaned up spacy-processed XML files
    This function is a generator that yields elements with the given tag from the XML file.
    """
    fp.seek(0)

    elements = enumerate(islice(etree.iterparse(fp, tag=tag), max_elements))
    # iterparse iterates through fp and only gets elements with the desired tag
    # islice allows us to iterate through the parsed list with step=max_elements (so we only get max_elements) and is more efficient because it yields elements one by one rather than copying the list slice
    for i, (event, elem) in elements:
        yield elem # returns this particular value, but doesen't end it so we create a data stream
        elem.clear() # empty the element so that it doesn't take up memory
        if progress_message and (i % 1000 == 0):  # check if we should print a progress message every 1000 elements
            print(progress_message.format(i), file=sys.stderr, end='\r')
    if progress_message: print(file=sys.stderr)


def short_xml_parse(fp, tag, max_elements=None): 
    """ 
    Parses cleaned up spacy-processed XML files (but not very well)
    This function loads the entire XML file into memory and returns a list of elements with the given tag, this is less efficient as doing it iteratively.
    """
    elements = etree.parse(fp).findall(tag) # etree.parse laods the whole xml into memory which is not as efficent :(
    # and then we final all instances of the tag from the loaded file (slow, sad)
    N = max_elements if max_elements is not None else len(elements) # this first checks if file is empty, if it is not empty it will return the max_elements, if it is empty it will return the length of the elements
    return elements[:N]

#####################################################################
# PCLVocab
#####################################################################

class PCLVocab(): 

    def __init__(self, vocab_file, vocab_size, num_stop_words): 
        '''
        This is the constructor for the class, it initializes the vocabulary starting from the index num_stop_words and ending at the index vocab_size. If vocab_size is None, it will include all words after num_stop_words. If num_stop_words is None, it will include all words from the beginning of the file.
        '''
        start_index = 0 if num_stop_words is None else num_stop_words
        end_index = start_index + vocab_size if vocab_size is not None else None

        self._words = [w.strip() for w in islice(vocab_file, start_index, end_index)] # use islice to load in vocab_file from start to stop index and then removes whitespace for each word in the vocad
        self._dict = dict([(w, i) for (i, w) in enumerate(self._words)]) # maps word to the place it appears in the file (enumerate returns (index, element) tuples)


    
    def __len__(self): 
        '''Get the length of the vocabulary'''
        return len(self._dict) # number of words (not checked for uniqueness)
    
    def index_to_label(self, i): 
        '''Get the word at a given index in the vocabulary'''
        return self._words[i] # ith element (elementrs are words in files)
    
    def __getitem__(self, key):
        '''Get the index (place that it appears) of a given word in the vocabulary'''
        if key in self._dict: return self._dict[key]
        else: return None

#####################################################################
# PCLLabels
#####################################################################

class PCLLabels(ABC):
    def __init__(self): 
        '''Constructor for the PCLLabels class, initializes the labels and label list to None'''
        self.labels = None
        self._label_list = None

    def __getitem__(self, index):
        """ return the label at this index """
        return self._label_list[index]

    def process(self, label_file, max_instances=None):
        '''
        This function processes the label file and returns a list of labels for each instance in the file. It uses the _extract_label method to extract the label for each instance, and then maps the labels to their corresponding indices in the label list. If the labels have not been initialized yet, it creates a sorted list of unique labels and a dictionary mapping each label to its index.
        '''
        
        y_labeled = list(map(self._extract_label, do_xml_parse(label_file, 'example', max_elements=max_instances)))
            # call do_xml_parse on the labels, the tag we use is 'example' (so we only get files marked example, and we only load max_instances of them.
            # then gets labels (somewhow) for each element yeilded by the do_xml_parse data stream function
        if self.labels is None:
            self._label_list = sorted(set(y_labeled)) # get all labels and sort increasing (set function gets unique labels)
            self.labels = dict([(x,i) for (i,x) in enumerate(self._label_list)]) # maps label to the place it appears in the file (enumerate returns (index, element) tuples): this is a dictionary
            
        y = [self.labels[x] for x in y_labeled] # this is a list of the indices of the labels in the label list (the place it appears in the file) — a list formed from the dictionary of labels
        return y

    @abstractmethod  # this decorator indicates that this method must be implemented by any subclass of PCLLabels since that's an abstract class.       
    def _extract_label(self, example):
        """ Return the label for this instance """
        return "Unknown"

#####################################################################
# PCLFeatures
#####################################################################

class PCLFeatures(ABC): 
    def __init__(self, vocab):
        '''
        This is a constructor for the PCLFeatures class which init's the vocab and vectoriser as defaultly sparse
        '''
        self.initial_vocab = vocab
        self.vectorizer = DictVectorizer(sparse=True) # Turns lists of feature value mapping dicts into a sparse matrix representation of the features, which is more memory efficient for large datasets. To standardize data for ML/prediction. won't do anything until we apply the vectorizer later...

    def extract_text(self, example):
        '''
        This method converts the example input into a list of individual words,, cleaned up via removing html tags & lowercasing text. It returns this list of words.
        '''
        return unescape("".join([x for x in example.itertext()]).lower()).split() # itertext() gets all of the text context in example, then we join the list of all texts in example, lowercase, unescape to remove html tags, and split on space
    

    def process(self, data_file, max_instances=None):
        '''
        This function processes the data file and returns a feature matrix X and a list of ids for each instance in the file. It uses the _extract_features method to extract the features for each instance, and then uses the DictVectorizer to transform the list of feature counters into a standardized clean representation. If max_instances is None, it processes all instances in the file; otherwise, it processes up to max_instances instances.
        
        '''
        if max_instances == None:
            N = len([1 for example in do_xml_parse(data_file, 'example')])
        else:
            N = max_instances
        
        ids = []
        feature_counters = [] # start empty list
        for example in do_xml_parse(data_file, 'example', max_elements=N, progress_message="Example {}"):
            # use the do_xml_parse to get the yielded data stream of N max_elements that have the tag 'example'
            ids.append(example.get("id")) # get the id category from the xml for the yielded date
            features = self._extract_features(example) # call the extract_features function which gives a list
            feature_counters.append(Counter(features)) # gets a dict of how many times each feature appears (no info on id here)
        X = self.vectorizer.fit_transform(feature_counters) # get X using our vectorizer; this takes the list of dicts of counts and transforms features into columns/categories and counts into the entries; not having the feature gets a zero entry
        return X, ids # return feature matrix and ids 
        # theoretically nth id maps to nth row of X (features)

    @abstractmethod
    def _get_feature_name(self, i):
        """ Returns a human-readable name for the ith feature in the DictVectorizer's internal vocabulary """
        return "Unknown"

    @abstractmethod            
    def _extract_features(self, example):
        """ Returns a list of the features in the example """
        return []

    @abstractmethod        
    def _get_num_features(self):
        """ Return the total number of features """
        return -1

#####################################################################