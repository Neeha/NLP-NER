Implementation of a simple binary classifier that can determine whether a token is part of a person's name using various techniques for feature extraction, feature indexing, optimization, etc. The data used in this project is derived from the CoNLL 2003 Shared Task on Named Entity Recognition.

To run the project
python3 classifier_main.py --model CLASSIFIER

Feature length  train : (204567, 136547)
Feature length  dev : (51578, 136547)

===Train accuracy===
F1: 0.987607

===Dev accuracy===
F1: 0.907864
