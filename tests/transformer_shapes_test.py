from modules.transformer.transformer import Transformer


class TestTransformerShapes:
    def test_transformer_shapes(
            self,
            src_vocab_size,
            trg_vocab_size,
            D,
            num_heads,
            D_ff,
            num_layers,
            input_sequence,
            target_sequence,
            batch_size,
            target_sequence_length,
            input_sequence_length
    ):
        transformer = Transformer(
            src_vocab_size,
            trg_vocab_size,
            D,
            num_heads,
            D_ff,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        logits, encoder_attention_weights, masked_mha_attention_weights, decoder_attention_weights = transformer(
            input_sequence, target_sequence
        )
        assert logits.shape == (batch_size, target_sequence_length, trg_vocab_size)
        assert encoder_attention_weights.shape == (batch_size, num_heads, input_sequence_length, input_sequence_length)
        assert masked_mha_attention_weights.shape == (
            batch_size, num_heads, target_sequence_length, target_sequence_length
        )
        assert decoder_attention_weights.shape == (batch_size, num_heads, target_sequence_length, input_sequence_length)
