#!usr/bin/python
import torch



class Encoder(torch.nn.Module):

        def __init__(self, input_size, embedding_dim, hidden_units, dropout):
        
                super(Encoder, self).__init__()
                
                #define the number of hidden units
                self.hidden_units = hidden_units
                
                
                #initialize the encoder layers
                
                #-----------------> Embedding layer
                self.embedding = torch.nn.Embedding(input_size, embedding_dim)
                
                #-----------------> LSTM bidirectional to summarize future and past words
                self.lstm_bid = torch.nn.LSTM(embedding_dim, hidden_units, bidirectional=False)
                
                #-----------------> Summarization LSTM to catch the meaning of the sentence
                self.lstm_sum = torch.nn.LSTM(hidden_units*2, hidden_units)
                
                
                
        def forward(self, inputs):
        
                #print('--------------------------------- Encoder ---------------------------------')
        
        
                #----------------------------------------------------> Input
                #print('input:',inputs.size())
                #(sentence length, batch size)
                
                embedded = self.embedding(inputs)
                
                #------------------------------------------------------------> Embedded
                #print('embedded:',embedded.size())
                #(sentence length, batch size, embedding dimension)
                
                #------------------------------------------------------------------> LSTM bidirectional
                output_bid, hidden_bid_states = self.lstm_bid(embedded)
                
                hidden_bid_h = hidden_bid_states[0]
                
                hidden_bid_c = hidden_bid_states[1]
                
                #print('lstm bid:',output_bid.size())
                #output: (sentence length, batch size, hid_dim * {bibirectional})
                
                #print('lstm bib h:',hidden_bid_h.size())
                #(layers, batch size, hid dim)
                              
                #print('lstm bid c:',hidden_bid_c.size())
                #(layers, batch size, hid dim)                
                
                #--------------------------------------------------------------------------> LSTM summarization
                #output_sum, hidden_sum_states = self.lstm_sum(output_bid)
                
                #hidden_sum_h = hidden_sum_states[0]
                
                #hidden_sum_c = hidden_sum_states[1]
                
                #print('lstm sum:',output_sum.size())
                #(sentence length, batch size, hid_dim * {bibirectional})
                
                #print('lstm sum h:',hidden_sum_h.size())
                #(layers, batch size, hid dim)
                              
                #print('lstm sum c:',hidden_sum_c.size())
                #(layers, batch size, hid dim)
                
                
                return output_bid, hidden_bid_h, hidden_bid_c
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
