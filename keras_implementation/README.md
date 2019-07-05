## Bahdanau attention


```python
#-------------------------------------------------------------> Bahdanau Attention layer

#use the layer as follows

import bahdanau_attention as bahdanau_attention

#input 
#(batch_size, timesteps, features_per_timestep)
#(batch_size, features_per_timestep)
 
bahdanau_attention.BahdanauAttention(n_units, name='bahdanau_attention_layer')

context_vector, attention_weights = bahdanau_attention_layer([hidden_state, hidden_sequence])

#output 
#context vector: (batch_size, features_per_timestep)
#attention weights: (batch_size, timesteps, 1)
```

## Compile and Train the model

```python

model = evaluate_model.compile_train_seq2seq(nmt_model, encoder_input_train, target_output_train, encoder_input_test, target_output_test, loss, epochs, learning_rate, batch_size, dropout_lstm_encoder, dropout_lstm_decoder, dropout_layer, decay)

```

## Evaluate the model

```python

evaluate_model.model_speech_evaluation(trained_model, output_tokenizer, encoder_input_train, train, role='Train')

```
