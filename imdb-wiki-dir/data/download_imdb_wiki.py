########################################################################################
# Code is based on the LDS and FDS (https://arxiv.org/pdf/2102.09554.pdf) implementation
# from https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir 
# by Yuzhe Yang et al.
########################################################################################
import os
import wget

print("Downloading IMDB faces...")
imdb_file = "imdb_crop.tar"
wget.download("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar", out=imdb_file)
print("Downloading WIKI faces...")
wiki_file = "wiki_crop.tar"
wget.download("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar", out=wiki_file)
print("Extracting IMDB faces...")
os.system(f"tar -xvf {imdb_file} -C ./data")
print("Extracting WIKI faces...")
os.system(f"tar -xvf {wiki_file} -C ./data")
os.remove(imdb_file)
os.remove(wiki_file)
print("\nCompleted!")