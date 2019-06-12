
Ubuntu 18.04.2 LTS

install anaconda https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart

```bash

#create conda envirnment

conda create --name gionanide_env

#all requirements

conda install -c lukepfister scikits.cuda

conda install -c anaconda cudatoolkit

conda install -c anaconda numpy

conda install -c anaconda pandas

conda install -c anaconda scipy

conda install -c conda-forge scikit-surprise

conda install -c anaconda gensim

conda install -c anaconda nltk

conda install -c anaconda scikit-learn

conda install -c conda-forge matplotlib

conda install bokeh

conda install -c anaconda tensorflow-gpu

conda install -c anaconda keras-gpu

conda install -c anaconda pydot
```
