##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##



import argparse
import sys
from PCLDataReader import PCLLabels, PCLFeatures, PCLVocab
from sklearn import MultinomialNB, DummyClassifier
from sklearn.model_selection import cross_val_predict



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
       self.initial_vocab = self.extract_text(vocab)
       
    
    def _extract_features(self, example):
        """This function returns a list of words which are in the input vocabulary. Words NOT in the vocab are then ignored."""
        words = self.extract_text(example) # extract the text from the example and split it into a list of words
        filtered_words = []
        for word in words: # iterate through the list of words
            if word in self.initial_vocab: # check if the word is in the vocab
                filtered_words.append(word) # if it is, add it to the filtered list
        return filtered_words # return the list of words that are in the vocab 


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
    vocabulary = PCLVocab(args.vocabulary) # first create an instance of the vocab
    features = MyFeatures(vocabulary) # then get features out of the vocab
    binary_labels = BinaryLabels() # create an instance of binary & category labels
    category_labels = CategoryLabels()
    clf = sklearn.MultinomialNB() # create an instance of naive bayes classifier 
    feature_matrix = features.process(data_file=args.data_file) # process the features we found (yay!)
    target_matrix = binary_labels.process(data_file=args.data_file) # use the binary labels 
    
    if args.test_category:
        # If a test category is given, then you'll use all of the examples from that category as test data, keeping only the examples without that category as training data. 
        test_data = np.where(category_labels.process(data_file=args.data_file) == args.test_category) # test is examples with category
        train_data = np.where(category_labels.process(data_file=args.data_file) != args.test_category) # train is examples without
        # get predictions and probabilities for each example in the matching test category
        clf.fit(feature_matrix[train_data], target_matrix[train_data])
        predictions = clf.predict(feature_matrix[test_data])
        probabilities = clf.predict_proba(feature_matrix[test_data])
    elif args.xvalidate:
        # Now, we perform x-fold cross validation on the full data, getting predictions (and probabilities) for every example
        probabilities = cross_val_predict(clf, feature_matrix, target_matrix, cv=args.xvalidate, method='predict_proba')
        predictions = np.argmax(probabilities, axis=1)
    
    # write out one line in args.output_file for each prediction in this format: example_id\spredicted class, true or false\sprobability
    for i, (example_id, pred, prob) in enumerate(zip(range(len(predictions)), predictions, probabilities)):
        true_label = target_matrix[i]
        pred_str = binary_labels[pred]
        args.output_file.write(f"{example_id}\s{pred_str}\s{prob[pred]}\n")
        with open(args.output_file, 'a') as f:
            f.write(f"{example_id}\s{pred_str}\s{prob[pred]}\n")
        
        
        
        
        

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

    for fp in (args.output_file, args.training, args.labels, args.vocabulary): fp.close()

