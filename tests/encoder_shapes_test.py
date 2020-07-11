from modules.transformer.encoder import EncoderLayer, Encoder


class TestEncoderLayerShapes:
    def test_encoder_shapes(
            self,
            src_vocab_size,
            D,
            num_heads,
            D_ff,
            num_layers,
            input_sequence,
            src_mask_tensor,
            batch_size,
            input_sequence_length
    ):
        encoder = Encoder(
            src_vocab_size, D, num_heads, D_ff, num_layers=num_layers
        )
        encoder_output, encoder_attention_weights = encoder(input_sequence, src_mask_tensor)
        assert encoder_output.shape == (batch_size, input_sequence_length, D)
        assert encoder_attention_weights.shape == (batch_size, num_heads, input_sequence_length, input_sequence_length)

