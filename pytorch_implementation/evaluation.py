#!usr/bin/python
import torch


#select GPU
device = torch.cuda.set_device(0)

def train(encoder, decoder, encoder_input, decoder_input, target, encoder_optimizer, decoder_optimizer, criterion):


        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()


        loss = 0


        #---------------------------------------------------------------------------------------> Encoder
        #initialize a list to keep all the hidden states
        hidden_states = []
          
        
        #call the encoder for every token of the sentence for all the encoder inputs
        for token in encoder_input:


                #for every input we have three outputs
                encoder_output, encoder_h, encoder_c = encoder(token.unsqueeze(1))
                #encoder_output: lstm hidden state
                #encoder_h: lstm hidden state (same as above)
                #encoder_c: cell/memory state
                
                hidden_states.append(encoder_output)

        #print('\n')
        #stack all the hidden states to make one tensor with all the hidden states
        encoder_output_sequence = torch.stack(hidden_states).squeeze(1)
        #print('Encoder hidden states:',encoder_output_sequence.shape)
        #print('Encoder last hidden state:',encoder_h.shape)
        #print('Encoder cell/memory state:',encoder_c.shape)
        #print('\n')
        
        
        

        decoder_hidden = [encoder_h, encoder_c]
        
        #keep a list with all the attention weights
        attention_weights_list = []
        
        #keep decoders outputs
        decoder_outputs = []
        
        
        #now we have to iterate the target and the decoder input (recall that those two have the same length)
        for token_index, token in enumerate(target):

                #-------------------------------------------------------------------------------------------> Decoder
                
                #for every input to the decoder we have three outputs
                decoder_output, decoder_hidden, attention_weights = decoder(decoder_input[token_index].unsqueeze(1), decoder_hidden, encoder_output_sequence)
                #print('Decoder output:',decoder_output.shape)
                #print('Decoder hidden state h:',decoder_hidden[0].shape)
                #print('Decoder cell/memory state:',decoder_hidden[1].shape)
                
                
                decoder_outputs.append(decoder_output)
                
                
                attention_weights_list.append(attention_weights)   
                

                current_target = target[token_index].unsqueeze(0)
                

                loss += criterion(decoder_output, current_target)
        
                             
                
                
                #print(targe_output[number_of_token],decoder_input[number_of_token])
                
                
        #print('\n')
        #print('---------- Decoder output for one training sample ----------')
        predictions = torch.stack(decoder_outputs).squeeze(1)
        
        attention_weights_list = torch.stack(attention_weights_list).squeeze(2)
                
        attention_weights_list = attention_weights_list.squeeze(2)

        
        #print('Model predictions:',predictions.shape)
        #print('Model attention:',attentions.shape)

        #print('predictions',predictions.shape)
        #print('target',target.shape)
        
        #backpropagation
        loss.backward()
        
        
        #weight optimization
        encoder_optimizer.step()
        decoder_optimizer.step()

                
                
        return predictions, attention_weights_list, (loss.item() / target.size(0))
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            
                
        #break
