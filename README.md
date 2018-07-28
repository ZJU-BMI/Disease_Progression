# Chronic Disease Progression Research
Not finished
## Introduction
TBD
## Experiment
### Data Structure
### Hyper-parameter Search

### Model Evaluate
We offer two ways to evaluate the performance of model.
1. Tensorboard, Tensorboard records the accuracy, specificity, precision, recall, f1, and time deviation of training 
set (mini batch) and test set respectively.
2. CSV. Csv file records the accuracy, specificity, precision, recall, f1, hamming loss, coverage, ranking loss, 
average precision, macro auc, micro auc, time deviation of training set (mini batch) and test set respectively

Launch tensorboard: type 'tensorboard --logdir={root path}\model_evaluate\{time(millisecond)}' in cmd. Once TensorBoard 
is running, navigate web browser to localhost:6006 to view the result. more details in [link](https://www.tensorflow.org/guide/summaries_and_tensorboard)

Csv. Training result is saved in {root path}\model_evaluate\{time(millisecond)}\train while the test result is saved 
in {root path}\model_evaluate\{time(millisecond)}\test. 
## To Be Done
