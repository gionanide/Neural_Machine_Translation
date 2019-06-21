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

model = evaluate_model.compile_train_seq2seq(model, encoder_inputs_train, decoder_inputs_train, target_outputs_train, encoder_inputs_test, decoder_inputs_test, target_outputs_test)

```

## Evaluate the model

```python

evaluate_model.model_speech_evaluation(trained_model, output_tokenizer, encoder_inputs_test, test, role='Test')

```
