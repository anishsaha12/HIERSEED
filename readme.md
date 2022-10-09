# HierSeed

The source code used for **Seeded Hierarchical Clustering for Expert-Crafted Taxonomies**, published in findings of **EMNLP 2022**.

## Requirements
We use Python 3.8 or above. Before running, you need to first install the required packages by typing following commands:

```console
$ pip install -r requirements.txt
```

Also, be sure to download 'punkt' in python:
```
import nltk
nltk.download('punkt')
```
We also use the B-Cubed measure implemented [here](https://github.com/m-wiesner/BCUBED).


## Quick Start
To train the HierSeed model with a particular dataset (in the `data/{dataset}` folder):

1. Run the notebook `4_train` with the input files consisting of `text_embedding_file`, `seed_embeddings_file`, `remaining_indices_file`. More information on how to generate these are found below.
2. Select the best iteration as the desired model parameters to be saved for use, based on the training evaluaitons provided.
3. Run the notebook `5_test_set_evaluate` to obtain evaluation reqults with test set.

The HierSeed class object's variable `shdc.topics_points_hierarchical` contains the indices of documents for each taxonomy topic. Refer to `evaluateSHDC_wos.py` "Assign to DF" section for more details on assigning the documents to topics.


## Running on New Dataset
To execute on your own dataset:
1. Create a new directory for you data within the `data` directory.
2. Run the notebook `1a_get_bert_embeddings`, `1b_get_glove_embeddings` or `1c_get_fasttext_embeddings` depending the type of text embedding to use. You may also use your own embeddings. Save the embeddings of each document in the dataset in a line-by-line manner. Keep the document and line index mapping.
3. Create the taxonomy information file with the same format as in `wos.taxnomy`. Run the `2a_get_hierarchy` notebook.
4. Create seed document set by running `2b_split_seed_random` notebook, and get their embeddings by running `3_get_seed_embeddings`. Use the same embedding file from step 2 as input.
