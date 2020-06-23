import torch
import torch.nn as nn
import logging
from typing import Tuple
from modules.transformer.sublayers import MultiHeadAttention, LayerNormalization, PositionWiseFeedForward
from modules.transformer.sublayers import EmbeddingLayer, PositionalEncoding


logger = logging.getLogger(__name__)


class DecoderLayer(nn.Module):
    """
    This class implements one decoder layer as described in the paper. The layer is composed by a masked multi-head
    attention layer followed by a residual layer normalisation. Its output is then fed into a multi-head attention
    layer, with the peculiarity that it runs the attention between the source and target representations. This is done
    by taking the key and value tensors from the encoder, whereas the query tensor comes from the previous step in the
    decoding process. Its output is also followed by a residual layer normalization. This output it then fed into a
    position-wise feed forward module, also followed by a residual layer normalisation.
    This module implements the residual connections in the forward pass.
    """
    def __init__(self, num_heads: int, D: int, D_ff: int, device: str = 'cpu', dropout: float = 0.1) -> None:
        """
        :param num_heads: the number of attention heads. The paper uses 8 heads.
        :param D: the length of the input matrix (per token). For the first decoder layer, this is the embedding
               matrix. In the paper D = 512.
        :param D_ff: the output dimension of the first layer and input dimension of the second (final) layer, within
               the position-wise feed forward layer of the decoder layer. In the paper, D_ff = 2048.
        :param device: the device on which to run the computations. Default: 'cpu'.
        :param dropout: the dropout applied to the each sub-layer. Default: 0.1
        """
        assert D % 2 == 0, f"D must be an even number, got {D}. Please reset it accordingly."
        super().__init__()
        self.D = D
        self.masked_mha = MultiHeadAttention(num_heads, D, device=device, dropout=dropout)  # this is masked
        self.layer_norm_1 = LayerNormalization(D, dropout=dropout)
        self.mha = MultiHeadAttention(num_heads, D, device=device, dropout=dropout)
        self.layer_norm_2 = LayerNormalization(D, dropout=dropout)
        self.pfww = PositionWiseFeedForward(D, D_ff, dropout=dropout)
        self.layer_norm_3 = LayerNormalization(D, dropout=dropout)

    def forward(
            self,
            target_sequence: torch.Tensor,
            encoder_output: torch.Tensor,
            trg_mask: torch.Tensor,
            src_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implements the forward pass for one decoder layer.
        This implements the residual connections between the first normalization layer and the target sequence,
        between the second and the first normalization layers, and between the third and the second normalization
        layers.
        :param target_sequence: the output from the previous decoder layer, i.e. the input into the current decoder
               layer. For the first decoder layer, this is the output from the positional encoding layer.
               Dim: (batch_size, trg_sequence_length, D)
        :param encoder_output: the key and value tensors from the encoder. Dim: (batch_size, src_sequence_length, D)
        :param trg_mask: The mask tensor to be applied to the first masked multi-head attention layer. This should be
               True for the positions of the padding tokens, as well as in all the illegal positions (i.e. the future
               positions that have not been decoded yet (the decoder is unidirectional as we do not have access to
               future tokens). and False otherwise. Dim: (batch_size, trg_sequence_length, trg_sequence_length)
        :param src_mask:  The mask tensor to be applied to the second multi-head attention layer. This should be True
               for the positions of the padding tokens and False otherwise.
               Dim: (batch_size, src_sequence_length, src_sequence_length)
        :return: norm_3: The output of the decoder layer, after applying the residual connection and layer
                 normalization to the position-wise feed forward layer. Dim: (batch_size, trg_sequence_length, D)
                 masked_mha_attention_weights: the attention weights of the masked multi-head attention sublayer.
                 This is a rank-4 tensor, as the weights from multiple heads have been stacked.
                 Dim: (num_heads, batch_size, trg_sequence_length, trg_sequence_length)
                 decoder_attention_weights: the attention weights of the second multi-head attention sublayer.
                 This is a rank-4 tensor, as the weights from multiple heads have been stacked.
                 Dim: (batch_size, num_heads, trg_sequence_length, src_sequence_length)
        """
        assert target_sequence.shape[0] == encoder_output.shape[0] == src_mask.shape[0] == trg_mask.shape[0], \
            f"target_sequence, encoder_output, src_mask and trg_mask must have the same batch size. Got " \
            f"{target_sequence.shape[0]}, {encoder_output.shape[0]}, {src_mask.shape[0]} and {trg_mask.shape[0]} " \
            f"respectively instead."
        assert len(target_sequence.shape) == len(encoder_output.shape) == len(src_mask.shape) == len(trg_mask.shape) \
            == 3, f"Expected rank-3 tensor for target sequence, encoder output, and source and target masks, " \
            f"including batch size. Got rank-{len(target_sequence.shape)} for target sequence, " \
            f"rank-{len(encoder_output.shape)} for encoder output, rank-{len(src_mask.shape)} for source mask and " \
            f"rank-{len(trg_mask.shape)} for target mask instead. If using single-batch, consider adding " \
            f"'.unsqueeze(0)' to your tensors."
        assert target_sequence.shape[2] == encoder_output.shape[2] == self.D, \
            f"The last dimension of the target tensor and the encoder output must be equal to D={self.D}. Got " \
            f"{target_sequence.shape[2]} for target tensor and {encoder_output.shape[2]} for encoder output instead."
        assert target_sequence.shape[1] == trg_mask.shape[1] == trg_mask.shape[2], \
            f"The second dimensions of target_sequence and the second and third dimension of trg_mask must all be " \
            f"equal to each other (this is the target sequence length). Got {target_sequence.shape[1]}, " \
            f"{trg_mask.shape[1]} and {trg_mask.shape[2]} instead."
        assert encoder_output.shape[1] == src_mask.shape[2], \
            f"The second dimensions of encoder_output and the third dimension of src_mask must all be equal to each " \
            f"other (this is the input sequence length). Got {encoder_output.shape[1]} and {src_mask.shape[2]} instead."
        # src_mask.shape[1] can be 1, and will be broadcasted to be equal to src_mask.shape[2] during operations.
        logger.info(
            f"Implementing the forward pass for DecoderLayer. Batch size: {target_sequence.shape[0]}, input sequence "
            f"length: {encoder_output.shape[1]}, target sequence length: {target_sequence.shape[1]}, embedding/inner "
            f"layer dimension: {target_sequence.shape[2]}."
        )
        masked_mha, masked_mha_attention_weights = self.masked_mha(
            target_sequence, target_sequence, target_sequence, mask=trg_mask
        )
        norm_1 = self.layer_norm_1(masked_mha + target_sequence)
        mha, decoder_attention_weights = self.mha(norm_1, encoder_output, encoder_output, mask=src_mask)
        norm_2 = self.layer_norm_2(mha + norm_1)
        pfww = self.pfww(norm_2)
        norm_3 = self.layer_norm_3(pfww + norm_2)
        return norm_3, masked_mha_attention_weights, decoder_attention_weights


class Decoder(nn.Module):
    """
    This class implements the decoder architecture, which is comprised of N stacked decoder layers. The paper uses
    N = 6 identical layers. First, the target sequence is passed through the embedding layer and the positional encoder,
    and the output from this layer is then fed into the first decoder layer. Each decoder layer feeds its output into
    the next decoder layer. This class returns the output of the final decoder layer.
    """
    def __init__(
            self,
            trg_vocab_size: int,
            D: int,
            num_heads: int,
            D_ff: int,
            max_seq_length: int = 512,
            num_layers: int = 6,
            trg_pad_idx: int = 0,
            device: str = 'cpu',
            dropout: float = 0.1,
    ) -> None:
        """
        :param trg_vocab_size: the length of the vocabulary of the target sequence, i.e. the size of the dictionary
               of embeddings. This does not have to be the same vocabulary of the source/input sequence, especially
               if the two sequences are in different languages.
        :param D: the length of the input matrix (per token). For the first decoder layer, this is the embedding
               matrix. In the paper D = 512.
        :param num_heads: the number of attention heads. The paper uses 8 heads.
        :param D_ff: the output dimension of the first layer and input dimension of the second (final) layer, within
               the position-wise feed forward layer of each decoder layer. In the paper, D_ff = 2048.
        :param max_seq_length: The maximum length of a sentence which can be used in this model. If a sentence is longer
               it must be truncated (this module does not provide a feature for truncation, it's up to the user to
               ensure sentences are within this limit). Most sentences will be much shorter. Default: 512.
        :param num_layers: the number of decoder layers. Default: 6.
        :param trg_pad_idx: the index in the target vocabulary for the '<pad>' token. Default: 0.
        :param device: the device on which to run the computations. Default: 'cpu'.
        :param dropout: the dropout applied to the each sub-layer of each decoder layer. Default: 0.1
        """
        assert D % 2 == 0, f"D must be an even number, got {D}. Please reset it accordingly."
        super().__init__()
        self.D = D
        self.embeddings = EmbeddingLayer(trg_vocab_size, D, pad_idx=trg_pad_idx)
        self.positional_encoding = PositionalEncoding(D, max_seq_length, dropout)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(num_heads, D, D_ff, device, dropout) for _ in range(num_layers)]
        )

    def forward(self, target_sequence, encoder_output, trg_mask, src_mask):
        """
        Implements the forward pass for the decoder.
        :param target_sequence: the target sentences, each sentence represented by numbers corresponding to the index of
               the sentence's words in the vocabulary. This must be a LongTensor. Dim: (batch_size, trg_sequence_length)
        :param encoder_output: the key and value tensors from the encoder. Dim: (batch_size, src_sequence_length, D)
        :param trg_mask: The mask tensor to be applied to the first masked multi-head attention layer of each decoder
               layer. This should be True for the positions of the padding tokens, as well as in all the illegal
               positions, i.e. the future positions that have not been decoded yet (the decoder is unidirectional as
               we do not have access to future tokens) and False otherwise.
               Dim: (batch_size, trg_sequence_length, trg_sequence_length)
        :param src_mask:  The mask tensor to be applied to the second multi-head attention layer of each decoder layer.
               This should be True for the positions of the padding tokens and False otherwise.
               Dim: (batch_size, src_sequence_length, src_sequence_length)
        :return: decoder_output: the output of the final decoder layer. Dim: (batch_size, trg_sequence_length, D)
                 masked_mha_attention_weights: the attention weights of the masked multi-head attention sublayer of the
                 final decoder layer. This is a rank-4 tensor, as the weights from multiple heads have been stacked.
                 Dim: (num_heads, batch_size, trg_sequence_length, trg_sequence_length)
                 decoder_attention_weights: the attention weights of the second multi-head attention sublayer of the
                 final decoder layer. This is a rank-4 tensor, as the weights from multiple heads have been stacked.
                 Dim: (batch_size, num_heads, trg_sequence_length, src_sequence_length)
        """
        assert target_sequence.shape[0] == encoder_output.shape[0] == src_mask.shape[0] == trg_mask.shape[0], \
            f"target_sequence, encoder_output, src_mask and trg_mask must have the same batch size. Got " \
            f"{target_sequence.shape[0]}, {encoder_output.shape[0]}, {src_mask.shape[0]} and {trg_mask.shape[0]} " \
            f"respectively instead."
        assert len(target_sequence.shape) == 2, \
            f"Expected rank-2 tensor as target sequence, including batch size, got rank-{len(target_sequence.shape)} " \
            f"instead. If using single-batch, consider adding '.unsqueeze(0)' to your input sequence."
        assert len(encoder_output.shape) == len(src_mask.shape) == len(trg_mask.shape) == 3, \
            f"Expected rank-3 tensor for encoder output and source and target masks, including batch size. Got" \
            f" rank-{len(encoder_output.shape)} for encoder output, rank-{len(src_mask.shape)} for source mask and " \
            f"rank-{len(trg_mask.shape)} for target mask instead. If using single-batch, consider adding " \
            f"'.unsqueeze(0)' to your tensors."
        assert encoder_output.shape[2] == self.D, \
            f"The last dimension of encoder_output must be equal to D={self.D}. Got {encoder_output.shape[2]} instead."
        assert target_sequence.shape[1] == trg_mask.shape[1] == trg_mask.shape[2], \
            f"The second dimension of target_sequence and the second and third dimension of trg_mask must all be " \
            f"equal to each other (this is the target sequence length). Got {target_sequence.shape[1]}, " \
            f"{trg_mask.shape[1]} and {trg_mask.shape[2]} instead."
        assert encoder_output.shape[1] == src_mask.shape[2], \
            f"The second dimensions of encoder_output and the third dimension of src_mask must all be equal to each " \
            f"other (this is the input sequence length). Got {encoder_output.shape[1]} and {src_mask.shape[2]} instead."
        # src_mask.shape[1] can be 1, and will be broadcasted to be equal to src_mask.shape[2] during operations.
        logger.info(
            f"Implementing the forward pass for Decoder. Batch size: {target_sequence.shape[0]}, input sequence "
            f"length: {encoder_output.shape[1]}, target sequence length: {target_sequence.shape[1]}, embedding "
            f"dimension: {encoder_output.shape[2]}."
        )

        embs = self.embeddings(target_sequence)
        decoder_output = self.positional_encoding(embs)
        for decoder in self.decoder_layers:
            decoder_output, masked_mha_attention_weights, decoder_attention_weights = decoder(
                decoder_output, encoder_output, trg_mask, src_mask
            )
        return decoder_output, masked_mha_attention_weights, decoder_attention_weights
