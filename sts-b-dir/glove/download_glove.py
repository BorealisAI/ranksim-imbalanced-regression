########################################################################################
# Code is based on the LDS and FDS (https://arxiv.org/pdf/2102.09554.pdf) implementation
# from https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir 
# by Yuzhe Yang et al.
########################################################################################
import os
import wget
import zipfile

print("Downloading and extracting GloVe word embeddings...")
data_file = "./glove/glove.840B.300d.zip"
wget.download("http://nlp.stanford.edu/data/glove.840B.300d.zip", out=data_file)
with zipfile.ZipFile(data_file) as zip_ref:
    zip_ref.extractall('/shared-data/imb-reg/glove')
os.remove(data_file)
print("\nCompleted!")