##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##



import argparse
import sys
import numpy as np
from PCLDataReader import PCLLabels, PCLFeatures, PCLVocab
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction import DictVectorizer



class BinaryLabels(PCLLabels):
    """This class is a subclass of PCLLabels that extracts binary labels from the XML data."""

    def _extract_label(self, example):
        """This function extracts the condescension attribute stored in an example taken from the ground-truth XML file. The condescension attribute is stored as the string "true" or "false", so that's what we return after extracting it from the example."""
        return example.get("condescension")


class CategoryLabels(PCLLabels):
    """This class is a subclass of PCLLabels that extracts category labels from the XML data."""

    def _extract_label(self, example):
        """This function extracts the category attribute stored in an example taken from the ground-truth XML file."""
        return example.get("category")
        
class MyFeatures(PCLFeatures):
    """This class is a subclass of PCLFeatures that extracts features from the XML data."""
    
    def __init__(self, vocab):
        """This is a constructor for the MyFeatures class which init's the vocab"""
        self.initial_vocab = vocab._words
        self.vectorizer = DictVectorizer()
       
    def check_upper(self, words):
        """Check if any word is entirely uppercase."""
        for word in words:
            if word.isupper():
                return True
        return False


    
        
        
    def _extract_features(self, example, extra_features=True, bigram=False):
        """This function returns a list of words which are in the input vocabulary. Words NOT in the vocab are then ignored."""
        words = self.extract_text(example) # extract the text from the example and split it into a list of words
        words_out = []
        contains_uppercase = False

        if (extra_features and self.check_upper(words)):
            contains_uppercase = True
        for word in words: # iterate through the list of words
            if word.lower() in self.initial_vocab: # check if the word is in the vocab
                # if (extra_features and word == "n't"):
                #     words_out.append("not")
                # else:
                words_out.append(word.lower()) # if it is, add it to the filtered list

        if bigram:
            bigrams = list(zip(words_out[:-1], words_out[1:]))
            words_out = []
            for bigram in bigrams:
                words_out.append(" ".join(bigram))
            
        if extra_features and contains_uppercase:
            words_out.append("CONTAINS_UPPERCASE")

        return words_out # return the list of words/bigrams that are in the vocab 
    
       


    def _get_feature_name(self, i):
        """ Returns a human-readable name for the ith feature in the DictVectorizer's internal vocabulary """
        word = self.vectorizer.feature_names_[i]
        return f"Count of {word}"
 
    def _get_num_features(self):
        """ Return the total number of features """
        
        return len(self.vectorizer.feature_names_)
    
    def bag_of_words_features(self, data_file, max_instances=None):
        '''
        Calls process with each indivdual word as its own feature
        Because thats how we wrote _extract_features :)
        '''
        data_file = "patronize_full.xml"
        self.process(data_file, max_instances=max_instances)

def do_experiment(args): 
    """
    Do the experiment!
    """
    vocabulary = PCLVocab(args.vocabulary, args.vocab_size, args.stop_words) # first create an instance of the vocab
    features = MyFeatures(vocabulary) # then get features out of the vocab
    binary_labels = BinaryLabels() # create an instance of binary & category labels
    category_labels = CategoryLabels()
    clf = MultinomialNB() # create an instance of naive bayes classifier 
    (feature_matrix, example_ids) = features.process(data_file=args.data_file)# process the features we found (yay!)
    args.data_file.seek(0)
    target_matrix =np.asarray(binary_labels.process(label_file=args.data_file)) # use the binary labels 
    
    if args.test_category:
        # If a test category is given, then you'll use all of the examples from that category as test data, keeping only the examples without that category as training data. 
        args.data_file.seek(0) # error fix thing as suggested in wesbite
        labels = np.asarray(category_labels.process(label_file=args.data_file))# convert labels to 1d array 
        test_cat_numerical = category_labels.labels[args.test_category]
        test_idx = np.where(labels == test_cat_numerical)[0] # check if we are in the tested category (if so, we're test!)
        train_idx = np.where(labels != test_cat_numerical)[0] # otherwise, we're train!

        feature_matrix = feature_matrix.tocsr() # convert to csr matrix format for fast row indexing (we saw on the sklearn sparse page that this lets us index rows faster, which is nice for filtering to rwos in train_idx)
    
        clf.fit(feature_matrix[train_idx], target_matrix[train_idx]) # filter out the test category from the training data and run the fitter on it
        predictions = clf.predict(feature_matrix[test_idx])
        probabilities = clf.predict_proba(feature_matrix[test_idx])
        output_indices = test_idx
      
        
        
    elif args.xvalidate:
        # Now, we perform x-fold cross validation on the full data, getting predictions (and probabilities) for every example
        probabilities = cross_val_predict(clf, feature_matrix, target_matrix, cv=args.xvalidate, method='predict_proba')
        predictions = np.argmax(probabilities, axis=1)
    
    # write out one line in args.output_file for each prediction in this format: example_id\spredicted class, true or false\sprobability
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        true_label = target_matrix[i]
        pred_str = binary_labels[pred]
        example_id = example_ids[i]
        # args.output_file.write(f"{example_id} {pred_str} {prob[pred]}\n")
        args.output_file.write(f"{example_id} {pred_str} {prob[pred]}\n")
        
        
        
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=argparse.FileType('rb'), help="Data file containing labeled training instances")
    parser.add_argument("vocabulary", type=argparse.FileType('r'), help="File containing vocabulary words")
    parser.add_argument("-o", "--output_file", type=argparse.FileType('w'), default=sys.stdout, help="Write predictions to FILE", metavar="FILE")
    parser.add_argument("-v", "--vocab_size", type=int, metavar="N", help="Only count the top N words from the vocab file", default=None)
    parser.add_argument("-s", "--stop_words", type=int, metavar="N", help="Exclude the top N words as stop words", default=None)
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances. N=0 means use all training instances.", default=None)

    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("-t", "--test_category")
    eval_group.add_argument("-x", "--xvalidate", type=int)

    args = parser.parse_args()
    do_experiment(args)
    for fp in (args.output_file, args.vocabulary): fp.close()
    #for fp in (args.output_file, args.training, args.labels, args.vocabulary): fp.close()

