# Sentence Boundary Detector
*author: Ivan Garrido Marquez*
# Introduction
This code implements a sentence boundary detector using machine learning. It includes all the programs produced to: generate training sets from a reference corpus, train a classifier with them, the classes and funtion to handle the sentence boundary detector and to evaluate them. Pre-trained models and testing files are also included. The section "Files" of this document deeply describes the contents of the respository and its organization. The section User Guide presents the instructions to use the programs.

# Technologies
The code was implemented with the following technologies:

 - Python 3.4.3
 - nltk 3.4.5
 - sklearn 0.20.4
 - numpy 1.15.1
 - scipy 0.13.3

Python 3 was the base language chosen to implement these programs. The nltk package provided a tokenizer, Part-Of-Speech tagger and the access to the Brown corpus. Sklearn provided the supervised learning algorithms employed in the method in this case the Decision Tree and the Multi-Layer Perceptron Neural Network classifiers. From Numpy we exploited its arrays and array operations. Scipy was not explicitly used in this code; however scipy sparse matrices are needed to train sklearn algorithms.

# User Guide
We run any these programs as any python program. We need to call the python3 interpreter in the command line. Some of these programs need arguments or can get optional arguments.
```sh
$ python3 program_name.py [options][parameters]
```
The help command line help of the following described programs can be read with the -h / -help option.

##### Generating training sets from corpus
To generate a training set ready for a machine learning algorithm for sentence boundary detection we use the preparing_training_set.py program. This program vectorizes the all the tokens in a set of sentences from the annotated brown corpus (as included in the nltk). Each vector represents the context of a given token. The context is given by the surrounding tokens. The final output this program produces is a matrix to serve as training set for a machine learining algorithm.

It gets as parameters:
- The starting integer index from which we will take the training documents from the brown corpus. Default = 0.
- The number of documents for the training set. Default = 500 (all the brown corpus).
- The size of the window for the context por representing the tokens. It represents the number of tokens we will take before and after the evaluated token. Default = 2.
- Directory path to store the output files. Default='.'. If the path does not exists it will try to create it.

##### Training a chosen classifier
The program train_classifier.py generates supervised classification models from a training data matrix. It gets as positional arguments:
- A file containing a pickle dumped matrix with a vectorized example per line to train the classifier.
- A file containing a pickle dumped list of the corresponding labels to each line in the vectorized example file.
- Directory path to store the output files. The out files are trained classification models. One with a learned decision tree classifier and the second with a MLP Neural Network.

The input files for the two first parameters (matrix and labels) must be produced with the preparing_training_set.py program.

##### Splitting a text into sentences
The program text_to_sentences.py splits an input text file into a list of sentences. Its arguments:
- The path and input text file to be split in sentences.
- A file with a sentence boundary detection classification model produced with the previously explained program train_classifier.py.
- The path and name for the output text file containing one sentence per line from the input file. If this parameter is not set the output will be sent to the standard output of the system.

##### API
The file tools_for_SBD.py includes some general purpose functions to make this approach work.
- A function to map Universal POS tags to integers
- The function to vectorize a certain token from a text in a position given by an index. The text must be already tokenized and pos tagged. The size of the context window must also be provided.
- A function used only for the evaluation. It returns the indices where we find sentence boundary tokens from a texts already divided in a list of sentences.
- A function to get the positions delimitating a sentence based on the indices computed by the previous function.

The file sentence_boundary_detector.py contains a class representing a sentence boundary detector. Its constructor requires the a pickle file with a classification model trained to detect sentence boundaries and the size of the context window correspondng the configuration of the training set. This class has two methods both of them get as argument a text document and return a list of sentences and a list of the positions of the detected boundaries. The difference betwwen them is that the one with the sufix test was designed to work with the annotated the corpus used to evaluate the approach. The other method *detectSentencesBoundaries()* is the one that should be employed in real practice for any non-corpus text.

##### Performance evaluation
evaluation_10FoldxValidation_sbd.py is a program made to automatize the evaluation by 10-fold cross-validation. It considerates a fixed scheme of 10 previously learned classifiers and 10 files containing vectorized sets of of tokens from a different sets of documents.

# Method
These programs are based and inspired by the approach presented in the article "Adaptive Multilingual Sentence Boundary Disambiguation" by David D. Palmer, Marti A. Hearst in 1997 ([PDF from ACL anthology][aclantho]). Although it can be seen as a direct implementation of that work some changes were added to simplify the method and the implementation. The following parts explain how it works and the differences with the original work.

##### Algorithm
The approach is simple, the main idea of the method is to see each token inside a text as a potential end/boundary of a sentence. A target text is sequentially scrolled token by token. Every token in a target text is labeled by a chosen supervised classifier. This classifier was trained with corpus data to learn to decide if a token is a sentence boundary or not. A sentence is built by adding sequentially all those tokens not labeled as boundaries by the classifier. When the classifier labels a token in the sequence as a boundary a new sentence is completely built.

##### Token representation
A token *t* is represented with its context, that is, the tokens surrounding *t*. The context is given by the *k* tokens occurring before and after *t*. All tokens are mapped to their Universal Part-Of-Speech types. In the practice, the tokens are represented as a vector of features of length *2k+3*. Being *2k* the size of the context window. The other 3 elements are: the POS type of the represented token itself; a flag saying if the following token starts with uppercase and a flag indicating if the token is among the punctuations '.','?','!'. Finally, the Universal POS tags are mapped to integers to have a numerical vector representation.

In the original algorithm in the base paper, instead of representing the context tokens as a single POS tag, it is a vector itself. They used a lexicon to determine the probabilities of every universal POS tag for each token. As I did not have access to such lexicon I trusted a POS tagger for simplicity. They added also 2 similar flags to the token representation, but different. The first flag says if the token itself starts with an uppercase. The second flag says if the next token is a punctuation.

##### Supervised classification algorithms
This implementation tests the same classification algorithms tested by the base paper. The chosen classifiers are Decision Tree and Multi-Layer Perceptron Neural Network. I did not implemented those algorithms, I used the implementations in the sklearn pakage instead. For simplicity, the classifiers were trained with the default parameters. A new version of this might have considered to estimate a set of parameters to provide a more optimal performance.

# Corpus and Dataset
The Brown corpus was used to train and test the approach. It was chosen because of the following reasons:
- It is already annotated with accurate POS tags
- It is already split in sentences, which made possible to extract reliable examples for a labeled training set.
- The base paper also employed this corpus, making possible the comparison.

The Brown corpus contains 500 documents of different thematic categories. The annotated version of the corpus was downloaded via the nltk package downloading tool and processed with the proper nltk functions.

# Performance evaluation

##### Testing methodology
10-fold cross-validation was performed over the brown corpus to test the approach.
For simplicity, the 500 documents corpus was divided sequentially into 10 subsets of 50 documents. Random sampling could be a good option to generate the folds but it will be taken into account maybe for a future experiment.

##### Results
The performance was evaluated by counting the rates of correct and incorrect detected boundaries in 4 standard metrics accuracy, precision, recall and f1-measure. The following table presents the evaluation results for the MLP Neural Network and the Decision Tree classifier.

| Metric | MLP | Tree |
| ------ | ------ | ------ |
| Accuracy | 0.9934 | 0.9926 |
| Precision | 0.9024 | 0.9080 |
| Recall | 0.9540 | 0.9327 |
| F1-measure | 0.9273 | 0.9200 |

# Files
All the programs (python code and shell scripts) are in the root of this repository. The main ones are explained in the user guide section.

The rest of the python files were small programs I made during the development process in order to test fractions of the code in an effective way. I decided to left them there because they can be useful for testing the approach in a way that can be easily understood.

The file "problem description" has the description of the task.
the file *test_file-the_strange_case_of_dr_jekyll_and_mr_hyde.txt* is the book of the same title, it can be used for testing. *sentences.output.txt* is the output file after spliting the sentences of the example text.

The folder classifiers has inside the 10 classifiers trained for the 10-fold cross-validation. The folder evaluation hols the files with the metrics computed in the performance evaluation. The folder training_logs has a file per each trained classifier with some information I printed out to verify the process. Finally, the training_set folder keeps all the training sets I generated for the 10-fold cross-validation.

# Further comments
In the first tests I was only using the POS context features. I wanted to observe how good this lexic patterns were able to capture the sentence boundaries. I wasn't working so well. I decided to finally add the features to explicitely indicate the punctuations . ! ? and the uppercase one. I think those features are highly important and they probably almost "do the magic". A feature section test could help to clarify this. 

I personally believe that a rule-based algorithm would perform ok for this problem. A very simple machine learning approach like the one implemented here proved to be reasonably good. A more complex machine learning approach might certainly improve the performance even more.

   [aclantho]: <https://www.aclweb.org/anthology/J97-2002/>
