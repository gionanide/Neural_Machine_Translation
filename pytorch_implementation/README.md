## ---------- Encoder ----------


encoder = encoder.Encoder(input_size, embedding_dim, hidden_units, dropout)

## ---------- Bahdanau Attention ----------


attention = bahdanau_attention.BahdanauAttention(units)

## ---------- Dencoder ----------


dencoder = decoder.Decoder(hidden_units, output_size, dropout_percent)
