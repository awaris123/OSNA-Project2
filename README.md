# OSNA-Project2

## To run the best performing model run GPUcolab_train_eval.ipynb
Click where it says "Open in Colab", this model requires the use of a GPU and if you are the TA or Professor, it will involve our shared drive. In fact cell 2 will ask to mount on your Google Drive, please be sure to follow the prompt. Then skip to the section titled "To run simple classifier on sentenceBERT embeddings" to train the model.

Run through all the cells, EXCEPT for the cells under the section headers: 

`This is the code modified from preprocessing.py to extract embeddings from BERT or even word2vec since these vectors are wasteful to recalculate on the fly. Don't run unless you want to extract and store embeddings.`

`Evaluate here... Do not unless it is for Kaggle Submission. Takes a bit of time`

These cells are not neccesary to run the main model and take a while because they are generating the embeddings on-the-fly. Save some compute power :) 

### Contact arai4 [at] hawk [dot] iit [dot] edu for access to the pretrained embeddings. Else you can run the code yourself to get the sbert embeddings. Use the preprocessing class in generate_embeddings.py to extract word2vec embeddings.
