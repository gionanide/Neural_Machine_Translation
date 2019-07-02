#!usr/bin/python
import torch
import bahdanau_attention_pytorch as bahdanau_attention


class Decoder(torch.nn.Module):


        def __init__(self, hidden_units, output_size, dropout_percent):
        
                super(Decoder, self).__init__()

                #initialize the properties
                self.hidden_units = hidden_units

                #range of Linear layer at the output
                self.output_size = output_size
                
                self.dropout_precent = dropout_percent
                
                
                #initialize layers

                #because we are using teacher forcing, we embed the input in the decoder (which is 1 timestep before the target)
                self.embedding = torch.nn.Embedding(self.output_size, hidden_units)
                
                
                self.dropout = torch.nn.Dropout(self.dropout_precent)

                
                self.lstm = torch.nn.LSTM(hidden_units, hidden_units)
                
                
                self.output = torch.nn.Linear(hidden_units, output_size)
                
                
                self.attn_combine = torch.nn.Linear(self.hidden_units * 2, self.hidden_units)


                self.softmax = torch.nn.LogSoftmax(dim=1)
                
                
                
                
        def forward(self, inputs, hidden_states, encoder_output_sequence):


                #print('--------------------------------- Decoder ---------------------------------')
                
                #---------------------------------------------------> Input
                #print('decoder input:',inputs.size())

                #-------------------------------------------> Embedding
                embedded = self.embedding(inputs)
                #print('embedded:',embedded.size())

                #----------------------------------------------> Relu activation
                output = self.dropout(embedded)
                #print('relu:',output.size())

                #-----------------------------------------------------------> LSTM 
                output, hidden = self.lstm(context_vector, hidden_states)
                #print('output:',output.size())
                #print(output[0].shape)

                #----------------------------------------------------------------------> Output
                output = torch.nn.functional.log_softmax(self.output(output[0]), dim=1)
                #print('softmax:',output.size())


                return output, hidden, attention_weights


        #in case of weights initialization
        def initHidden(self):


               return torch.zeros(1, 1, self.hidden_units, device=device)
                
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
