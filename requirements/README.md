Ubuntu 18.04.2 LTS

install python: sudo apt-get install python3.6

install anaconda: https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart


```bash

#create conda envirnment -- conda version: 4.6.12

conda create --name conda_env

#all requirements

conda install -c lukepfister scikits.cuda==0.5.2

conda install -c anaconda cudatoolkit==10.1.168

conda install -c anaconda numpy==1.16.4

conda install -c anaconda pandas==0.24.2

conda install -c anaconda scipy==1.2.1

conda install -c conda-forge scikit-surprise==1.0.6

conda install -c anaconda gensim==3.4.0

conda install -c anaconda nltk==3.4.1

conda install -c anaconda scikit-learn==0.21.2

conda install -c conda-forge matplotlib==3.1.0

conda install bokeh==1.2.0

conda install -c anaconda tensorflow-gpu==1.13.1

conda install -c anaconda keras-gpu==2.2.4

conda install -c anaconda pydot==1.4.1
