import torch
import torch.nn as nn
from typing import Tuple
import logging
from modules.transformer.sublayers import MultiHeadAttention, LayerNormalization, PositionWiseFeedForward
from modules.transformer.sublayers import EmbeddingLayer, PositionalEncoding


logger = logging.getLogger(__name__)


class EncoderLayer(nn.Module):
    """
    This class implements one encoder layer as described in the paper. The layer is composed by a multi-head attention
    layer followed by a residual layer normalisation. Its output is then fed into a position-wise feed forward module,
    also followed by a residual layer normalisation.
    This module implements the residual connections in the forward pass.
    """
    def __init__(self, num_heads: int, D: int, D_ff: int, device: str = 'cpu', dropout: float = 0.1) -> None:
        """
        :param num_heads: the number of attention heads. The paper uses 8 heads.
        :param D: the length of the input matrix (per token). For the first encoder layer, this is the embedding
               matrix. In the paper D = 512.
        :param D_ff: the output dimension of the first layer and input dimension of the second (final) layer, within
               the position-wise feed forward layer of the encoder layer. In the paper, D_ff = 2048.
        :param device: the device on which to run the computations. Default: 'cpu'.
        :param dropout: the dropout applied to the each sub-layer. Default: 0.1  # WORTH HAVING A LAYER-BY-LAYER DROPOUT?
        """
        assert D % 2 == 0, f"D must be an even number, got {D}. Please reset it accordingly."
        super().__init__()
        self.D = D
        self.mha = MultiHeadAttention(num_heads, D, device=device, dropout=dropout)
        self.layer_norm_1 = LayerNormalization(D, dropout=dropout)
        self.pwff = PositionWiseFeedForward(D, D_ff, dropout=dropout)
        self.layer_norm_2 = LayerNormalization(D, dropout=dropout)

    def forward(self, input_sequence: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the forward pass for one encoder layer.
        This implements the residual connections between the first normalization layer and the input sequence, and
        between the second and the first normalization layers.
        :param input_sequence: the output from the previous encoder layer, i.e. the input into the current encoder
               layer. For the first encoder layer, this is the output from the positional encoding layer.
               Dim: (batch_size, src_sequence_length, D)  # NEED TO CHECK WHETHER WE NEED 3 SEPARATE ONES.
        :param src_mask: The mask tensor to be applied to the multi-head attention layer. This should be True for
               the positions of the padding tokens and False otherwise.
               Dim: (batch_size, src_sequence_length, src_sequence_length)
        :return: norm_2: The output of the encoder layer, after applying the residual connection and layer
                 normalization to the position-wise feed forward layer. Dim: (batch_size, src_sequence_length, D)
                 encoder_attention_weights: the attention weights of an individual encoder layer. This is a rank-4
                 tensor, as the weights from multiple heads have been stacked.
                 Dim: (batch_size, num_heads, src_sequence_length, src_sequence_length)
        """
        assert input_sequence.shape[0] == src_mask.shape[0], \
            f"input_sequence and src_mask must have the same batch size. Got {input_sequence.shape[0]} and " \
            f"{src_mask.shape[0]} instead."
        assert len(input_sequence.shape) == len(src_mask.shape) == 3, \
            f"Expected rank-3 tensor for input sequence and mask, including batch size, got " \
            f"rank-{len(input_sequence.shape)} for input sequence, and rank-{len(src_mask.shape)} for mask instead. " \
            f"If using single-batch, consider adding '.unsqueeze(0)' to your tensors."
        assert input_sequence.shape[2] == self.D, \
            f"The last dimension of the input tensor must be equal to D={self.D}.Got {input_sequence.shape[2]} instead."
        assert input_sequence.shape[1] == src_mask.shape[2], \
            f"The second dimensions of input_sequence and the third dimension of src_mask must all be equal to each " \
            f"other (this is the input sequence length). Got {input_sequence.shape[1]} and {src_mask.shape[2]} instead."
        # src_mask.shape[1] can be 1, and will be broadcasted to be equal to src_mask.shape[2] during operations.
        logger.info(
            f"Implementing the forward pass for EncoderLayer. Batch size: {input_sequence.shape[0]}, input sequence "
            f"length: {input_sequence.shape[1]}, embedding/inner layer dimension: {input_sequence.shape[2]}."
        )
        mha, encoder_attention_weights = self.mha(input_sequence, input_sequence, input_sequence, mask=src_mask)
        norm_1 = self.layer_norm_1(mha + input_sequence)  # residual connection with added input_sequence
        pwff = self.pwff(norm_1)
        norm_2 = self.layer_norm_2(pwff + norm_1)  # residual connection with added norm_1
        return norm_2, encoder_attention_weights


class Encoder(nn.Module):
    """
    This class implements the encoder architecture, which is comprised of N stacked encoder layers. The paper uses
    N = 6 identical layers. First, the input sequence is passed through the embedding layer and the positional encoder,
    and the output from this layer is then fed into the first encoder layer. Each encoder layer feeds its output into
    the next encoder layer. This class returns the output of the final encoder layer.
    """
    def __init__(
            self,
            src_vocab_size: int,
            D: int,
            num_heads: int,
            D_ff: int,
            max_seq_length: int = 512,
            num_layers: int = 6,
            src_pad_idx: int = 0,
            device: str = 'cpu',
            dropout: float = 0.1
    ) -> None:
        """
        :param src_vocab_size: the length of the vocabulary of the source sequence, i.e. the size of the dictionary
               of embeddings.
        :param D: the length of the input matrix (per token). For the first encoder layer, this is the embedding
               matrix. In the paper D = 512.
        :param num_heads: the number of attention heads. The paper uses 8 heads.
        :param D_ff: the output dimension of the first layer and input dimension of the second (final) layer, within
               the position-wise feed forward layer of each encoder layer. In the paper, D_ff = 2048.
        :param max_seq_length: The maximum length of a sentence which can be used in this model. If a sentence is longer
               it must be truncated (this module does not provide a feature for truncation, it's up to the user to
               ensure sentences are within this limit). Most sentences will be much shorter. Default: 512.
        :param num_layers: the number of encoder layers. Default: 6.
        :param src_pad_idx: the index in the source vocabulary for the '<pad>' token. Default: 0.
        :param device: the device on which to run the computations. Default: 'cpu'.
        :param dropout: the dropout applied to the each sub-layer of each encoder layer. Default: 0.1
        """
        assert D % 2 == 0, f"D must be an even number, got {D}. Please reset it accordingly."
        super().__init__()
        self.D = D
        self.embeddings = EmbeddingLayer(src_vocab_size, D, pad_idx=src_pad_idx)
        self.positional_encoding = PositionalEncoding(D, max_seq_length, dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(num_heads, D, D_ff, device, dropout) for _ in range(num_layers)]
        )

    def forward(self, input_sequence: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the forward pass for the encoder.
        :param input_sequence: The input sentences, each sentence represented by numbers corresponding to the index of
               the sentence's words in the vocabulary. This must be a LongTensor. Dim: (batch_size, src_sequence_length)
        :param src_mask: The mask tensor to be applied to the multi-head attention layer of each encoder layer. This
               should be True for the positions of the padding tokens and False otherwise.
               Dim: (batch_size, src_sequence_length, src_sequence_length)
        :return: encoder_output: the output of the final encoder layer. Dim: (batch_size, src_sequence_length, D)
                 encoder_attention_weights: the attention weights of the final encoder layer. This is a rank-4 tensor,
                 as the weights from multiple heads have been stacked.
                 Dim: (batch_size, num_heads, src_sequence_length, src_sequence_length)
        """
        assert input_sequence.shape[0] == src_mask.shape[0], \
            f"input_sequence and src_mask must have the same batch size. Got {input_sequence.shape[0]} and " \
            f"{src_mask.shape[0]} instead."
        assert len(input_sequence.shape) == 2, \
            f"Expected rank-2 tensor as input sequence, including batch size, got rank-{len(input_sequence.shape)} " \
            f"instead. If using single-batch, consider adding '.unsqueeze(0)' to your input sequence."
        assert len(src_mask.shape) == 3, \
            f"Expected rank-3 tensor for source mask, including batch size, got rank-{len(src_mask.shape)} " \
            f"instead. If using single-batch, consider adding '.unsqueeze(0)' to your input sequence."
        assert input_sequence.shape[1] == src_mask.shape[2], \
            f"The second dimension of input_sequence and the third dimension of src_mask must all equal to each other" \
            f" (this is the input sequence length). Got {input_sequence.shape[1]}, and {src_mask.shape[2]} instead."
        # src_mask.shape[1] can be 1, and will be broadcasted to be equal to src_mask.shape[2] during operations.
        logger.info(
            f"Implementing the forward pass for Encoder. Batch size: {input_sequence.shape[0]}, input sequence length: "
            f"{input_sequence.shape[1]}, embedding dimension: {self.D}."
        )
        embs = self.embeddings(input_sequence)
        encoder_output = self.positional_encoding(embs)
        for encoder in self.encoder_layers:
            encoder_output, encoder_attention_weights = encoder(encoder_output, src_mask)
        return encoder_output, encoder_attention_weights
