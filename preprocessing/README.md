```python
'''

------ input ------

- the path of the file that you want to read


------ output ------

- return the context of the file as a string

'''
def readFile(path):






'''

------ input -----

- the context of the file as a string


------ output ------

- by reading every line seperately convert it from ---- Hi.	Hallo! ---- to ---- [Hi.,Hallo!] -- a list

'''
#------------------------------------------------------------------------------------> Preprocessing
def makePairs(text):   






'''

Cleaning prolicy:

---- remove all non printable chracters
---- remoce all the puncuation characters
---- make all the unicodes charaxters to ASCII
---- make all the letters lowercase
---- remove non alphabetic tokenss

a signle example 

------ input ------

- [Hi.,Hallo!]


------ output ------


- [hi,hallo], without teacher forcing
- [sos hi eos, sos hallo eos], with teacher forcing

'''
def cleaning(pairs,forcing):   






'''

A function just to visualize the pairs

e.g.  hi ------> hallo

'''
#check the results
def visualize(cleaned_words):






'''

Initialize a Tokenizer and fit it on the text

------ input ------

- the list containing all the cleaned pairs


------ output ------

creates a vocabulary that keeps the count of every word in the text, and a vocabulary with every words unique index, in our procedure we are using the index vocabulary to encode the input and after the prediction decode the predictions of our network
- count{'hi':10}
- index{'hi':4}

'''
#we user keras tokenizer to map words to integers
def tokenizer(pairs):  






'''

Here we want to calculate the length of the max sentence for each of the languages, recall the we have the constraint of static vector

'''
#find the length of the max sentence
def max_sentence_length(pairs):  






'''

We are now encoding the sequences as number to feed our network

------ input ------

- [hi,hallo]


------ output ------

- [[4],[6]], to see how it works it is like this [[hi],[hallo]] so we replace every word with it's index 

if I had more word for example in the teacher forcing procedure

- [[1,4,2],[1,6,2]], to see it [[sos,hi,eos],[sos,hallo,eos]], recall that every language has differente indexing

'''
#encode sequences
def encode_sequences(tokenizer, pairs):  






'''

Recall that we had the constraint of the static vector so we have to pad our sequences with respect to the max length, so to reach this length we add zeros until we make it

------ input ------

- [[4],[6]]


------ output ------

- [[4,0,0.....,0],[6,0,0,.....,0]], every language has different padding but it depends of what is the input and the output of your model for the padding procedure

'''
#apply zero padding to sequences
def pad_sequences(length, pairs):    






'''

Now we have to define the target of our model, we answer the question, What to output?

------ input ------

- [[4,0,0.....,0],[6,0,0,.....,0]], recall that this is the result of precessing of [[Hi.],[Hallo!]], after the previous preprocessing steps


------ output ------

now depend of the output, if I want to go english -> german, german -> english or both, but the procedure is the same only some properties changes, e.g. the lenght of the max sentence and the indexing

- e.g. we are going english -> german: we are keeping the english word as it is :[[4,0,0.....,0]
- but we have to vhange the german word which is the target. Recall that we said we have to do this as a matrix (size_of_max_sentence x vocabulary_size) [[0,0,0,0,0,1,0,0,0,0....0],[],[],......,[]]], see that the 1 is in the position 6, that is the index that I want to predict, all the other rows consist only from zeros because there is not another word to predict, so our network has to learn to output only zeros.

'''
#apply one hot encoding to the output
def oneHotEncoding(sequences, vocabulary_size):  






'''

Now we are making different format: if I do not want the teacher foarcing procedure I do not want to use them, but in the teacher forcing procedure I have to format the deocoder's input as well. To do this I have to remove the EOS symbol from it's inputs and to remove SOS symbol from it's output.

index{sos:1}
index{eos:2}

------ input ------

- [1,6,8,17,2,0,0,0,0,0]



------ output ------

- [6,8,17,2,0,0,0,0,0,0]

'''
#remove Start Of Sequence symbol 
def removeSOS(array,set_size): 






'''
        
------ input ------

- [1,6,8,17,2,0,0,0,0,0]



------ output ------

- [1,6,8,17,0,0,0,0,0,0]

'''
#remoce End Of Sequence sumbol       
def removeEOS(array):       






'''

In the following functions we are combine all the preprocessing functions, let's see the flow for just one pair, and say that we want translation from english to german, without using teacher forcing.

Also let's say: index{hi:5}, index{hallo:4}, english_max_sentence=7, german_max_sentence=6, german_vocabulary=9(consist of only 3 words)

------ input ------

- Hi.	Hallo!


------ output ------

[models_inputs,models_output]

- models inputs: [5,0,0,0,0,0,0]
- models output: [[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]

'''
def format_data(path, dataset_length, train_per_cent, translation_flow):      






'''

In the following functions we are combine all the preprocessing functions, let's see the flow for just one pair, and say that we want translation from english to german, with using teacher forcing.

Also let's say: index{hi:5}, index{hallo:4}, index{sos:1}, index{eos:2}, english_max_sentence=7, german_max_sentence=6, german_vocabulary=9(consist of only 3 words)

------ input ------

- Hi.	Hallo!


------ output ------

[models_inputs,models_output]

- models inputs: encoder_inputs:[5,2,0,0,0,0,0], decoder_inputs=[1,4,0,0,0,0,0,0,0]
- models output: [[0,0,0,1,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]

'''
#different format for teacher forcing procedure
def teacher_forcing_format(path, dataset_length, train_per_cent, translation_flow):     
```
        
        
        
        
        
        
        
        
        
        
     
        
     
    
    
           
