from modules.transformer.decoder import DecoderLayer, Decoder


class TestDecoderLayerShapes:
    def test_decoder_layer_shapes(
            self,
            num_heads,
            D,
            D_ff,
            target_sequence_rank3,
            encoder_output,
            trg_mask_tensor,
            src_mask_tensor,
            batch_size,
            target_sequence_length,
            input_sequence_length
    ):
        decoder_layer = DecoderLayer(
            num_heads,
            D,
            D_ff
        )
        norm_3, masked_mha_attention_weights, decoder_attention_weights = decoder_layer(
            target_sequence_rank3, encoder_output, trg_mask_tensor, src_mask_tensor
        )
        assert norm_3.shape == (batch_size, target_sequence_length, D)
        assert masked_mha_attention_weights.shape == (
            batch_size, num_heads, target_sequence_length, target_sequence_length
        )
        assert decoder_attention_weights.shape == (batch_size, num_heads, target_sequence_length, input_sequence_length)


class TestDecoderShapes:
    def test_decoder_shapes(
            self,
            trg_vocab_size,
            D,
            num_heads,
            D_ff,
            num_layers,
            target_sequence,
            encoder_output,
            trg_mask_tensor,
            src_mask_tensor,
            batch_size,
            target_sequence_length,
            input_sequence_length
    ):
        decoder = Decoder(
            trg_vocab_size, D, num_heads, D_ff, num_layers=num_layers
        )
        decoder_output, masked_mha_attention_weights, decoder_attention_weights = decoder(
            target_sequence, encoder_output, trg_mask_tensor, src_mask_tensor
        )
        assert decoder_output.shape == (batch_size, target_sequence_length, D)
        assert masked_mha_attention_weights.shape == (
            batch_size, num_heads, target_sequence_length, target_sequence_length
        )
        assert decoder_attention_weights.shape == (batch_size, num_heads, target_sequence_length, input_sequence_length)
