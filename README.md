[![Build Badge](https://github.com/andreasottana/deep_learning_models/workflows/shapes%20tests/badge.svg)](https://github.com/AndreaSottana/deep_learning_models/actions)

# Deep Learning Models

This repository includes a number of deep learning models written by myself using the PyTorch deep learning framework.
The models are included in the `modules` folder, whereas saved trained models are in the `models` folder. 
The `script` folder contains some readily available scripts to run to ensure the models work as expected.
This repository is currently a work in progress and more things will be added as I build them.  
 
 
The currently available models are:  

:white_check_mark: **Transformer**: the transformer model (including the Encoder, the Decoder and all respective 
sublayers) as described in the paper 
"<a target="_blank" href="https://arxiv.org/pdf/1706.03762.pdf">Attention is All You Need</a>", Dec 2017, written 
entirely in `pytorch` with no additional deep learning libraries.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:radio_button: Includes transformer training function for Machine 
Translation tasks  

:white_check_mark: **BERT for Question Answering Fine Tuning**: full code to fine tune
a base BERT model from `transformers.BertForQuestionAnswering` (i.e. a general, not-task-specific pre-trained BERT 
model) for question answering tasks using the well-known  
<a target="_blank" href="https://rajpurkar.github.io/SQuAD-explorer/">SQuAD v1.1 dataset</a>. The code includes all the
data pre-processing steps, the training (fine-tuning) and validation of the model, as well as the scores calculations 
based on the predictions on a test set. The code provides both CPU and cuda (GPU) support, and a final chat-bot 
interface to answer questions given a context and the trained model.