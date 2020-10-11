import torch
import torch.nn as nn
import logging
from typing import Tuple
from .decoder import Decoder
from .encoder import Encoder


logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    """
    This module implements the transformer model as described in the paper 'Attention is All You Need', Dec 2017,
    https://arxiv.org/pdf/1706.03762.pdf. It comprises of an encoder and a decoder, each constituted by a number of
    identical stacked encoder and decoder layers respectively. The paper uses 6 encoder layers and 6 decoder layers.
    At the end of the decoder, there is a linear layer followed by a softmax activation.
    This class also contains private methods to create source and target masks.

    The weights are initialized using the Xavier distribution. This will automatically initialize all the weights in
    the more granular blocks that build the Transformer (i.e. the Encoder, Decoder, and all their layers and sublayers)
    also with the same Xavier distribution. Networks initialized with Xavier achieved substantially quicker
    convergence and higher accuracy, see http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    def __init__(
            self,
            src_vocab_size: int,
            trg_vocab_size: int,
            D: int,
            num_heads: int,
            D_ff: int,
            max_seq_length: int = 512,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            src_pad_idx: int = 0,
            trg_pad_idx: int = 0,
            device: str = 'cpu',
            dropout: float = 0.1,
    ) -> None:
        """
        :param src_vocab_size: the length of the vocabulary of the source sequence, i.e. the size of the dictionary
               of embeddings.
        :param trg_vocab_size: the length of the vocabulary of the target sequence, i.e. the size of the dictionary
               of embeddings. This does not have to be the same vocabulary of the source/input sequence, especially
               if the two sequences are in different languages.
        :param D: the length of the input matrix (per token). For the first encoder and first decoder layer, this is
               the embedding matrix. In the paper D = 512.
        :param num_heads: the number of attention heads used in each (masked) multi-head attention layer.
               The paper uses 8 heads.
        :param D_ff: the output dimension of the first layer and input dimension of the second (final) layer, within
               the position-wise feed forward layer of each encoder and decoder layer. In the paper, D_ff = 2048.
        :param max_seq_length: The maximum length of a sentence which can be used in this model. If a sentence is longer
               it must be truncated (this module does not provide a feature for truncation, it's up to the user to
               ensure sentences are within this limit). Most sentences will be much shorter. Default: 512.
        :param num_encoder_layers: the number of encoder layers. Default: 6.
        :param num_decoder_layers: the number of decoder layers. Default: 6.
        :param src_pad_idx: the index in the source vocabulary for the '<pad>' token. Default: 0.
        :param trg_pad_idx: the index in the target vocabulary for the '<pad>' token. Default: 0.
        :param device: the device on which to run the computations. Default: 'cpu'.
        :param dropout: the dropout applied to the each sub-layer of each encoder and decoder layer. Default: 0.1
        """
        assert D % 2 == 0, f"D must be an even number, got {D}. Please reset it accordingly."
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.encoder = Encoder(
            src_vocab_size, D, num_heads, D_ff, max_seq_length, num_encoder_layers, src_pad_idx, device, dropout
        )
        self.decoder = Decoder(
            trg_vocab_size, D, num_heads, D_ff, max_seq_length, num_decoder_layers, trg_pad_idx, device, dropout
        )
        self.linear = nn.Linear(D, trg_vocab_size)

        # initializing all the model parameters using Xavier uniform distribution, see docstring above
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_src_mask(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        In the source mask, we mask anywhere there is a '<pad>' token.
        :param input_sequence:  the input sentences, each sentence represented by numbers corresponding to the index of
               the sentence's words in the source vocabulary. This must be a LongTensor.
               Dim: (batch_size, src_sequence_length)
        :return: src_mask: a tensor of booleans. True in those positions where there is a '<pad>' token in the
                 input_sequence, and False otherwise. Dim: (batch_size, 1, src_sequence_length)
                 IMPORTANT NOTE: The second dimension is 1. However, this will be broadcasted to be equal to
                 src_sequence_length or trg_sequence_length in downstream operations, accordingly.
        """
        src_mask = (input_sequence != self.src_pad_idx).unsqueeze(-2)
        return src_mask

    def _create_trg_mask(self, target_sequence: torch.Tensor) -> torch.Tensor:
        """
        In the target mask, we mask anywhere there is a '<pad>' token, as well as in all the illegal positions (i.e.
        the future positions that have not been decoded yet (the decoder is unidirectional as we do not have access to
        future tokens).
        :param target_sequence:  the target sentences, each sentence represented by numbers corresponding to the index
               of the sentence's words in the target vocabulary. This must be a LongTensor.
               Dim: (batch_size, trg_sequence_length)
        :return: trg_mask: a tensor of booleans. True in those positions where there is a '<pad>' token in the
                 target_sequence, as well as in all the illegal/future positions, and False otherwise.
                 Dim: (batch_size, trg_sequence_length, trg_sequence_length)
        """
        trg_mask = (target_sequence != self.trg_pad_idx).unsqueeze(-2)
        illegal_pos_mask = torch.ones((1, target_sequence.shape[1], target_sequence.shape[1])).triu(1) == 0
        trg_mask = trg_mask & illegal_pos_mask  # bitwise and operator
        # broadcasting turns (A, 1, C) & (1, B, C) into (A, B, C) shape, i.e. (batch_size, trg_seq_len, trg_seq_len)
        return trg_mask

    def forward(
            self, input_sequence: torch.Tensor, target_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implements the forward pass for the transformer.
        :param input_sequence: the input sentences, each sentence represented by numbers corresponding to the index of
               the sentence's words in the source vocabulary. This must be a LongTensor.
               Dim: (batch_size, src_sequence_length)
        :param target_sequence: the target sentences, each sentence represented by numbers corresponding to the index
               of the sentence's words in the target vocabulary. This must be a LongTensor.
               Dim: (batch_size, trg_sequence_length)
        :return: logits: the transformer output, after applying a linear layer to the decoder output.
                This linear layer converts the last dimension from D to trg_vocab_size.
                 Dim: (batch_size, trg_sequence_length, trg_vocab_size)
                 encoder_attention_weights: the attention weights of the final encoder layer. This is a rank-4 tensor,
                 as the weights from multiple heads have been stacked.
                 Dim: (batch_size, num_heads, src_sequence_length, src_sequence_length)
                 masked_mha_attention_weights: the attention weights of the masked multi-head attention sublayer of the
                 final decoder layer. This is a rank-4 tensor, as the weights from multiple heads have been stacked.
                 Dim: (batch_size, num_heads, trg_sequence_length, trg_sequence_length)
                 decoder_attention_weights: the attention weights of the second multi-head attention sublayer of the
                 final decoder layer. This is a rank-4 tensor, as the weights from multiple heads have been stacked.
                 Dim: (batch_size, num_heads, trg_sequence_length, src_sequence_length)
        """
        assert input_sequence.shape[0] == target_sequence.shape[0], \
            f"Input and target sequences must have the same batch size. Got {input_sequence.shape[0]} and " \
            f"{target_sequence.shape[0]} instead."
        assert len(input_sequence.shape) == len(target_sequence.shape) == 2, \
            f"Expected rank-2 tensor for input and target sequences, including batch size. Got " \
            f"rank-{len(input_sequence.shape)} for input sequence and rank-{len(target_sequence.shape)} for target " \
            f"sequence instead. If using single-batch, consider adding '.unsqueeze(0)' to your tensors."
        logger.info(
            f"Implementing the forward pass for Transformer. Batch size: {input_sequence.shape[0]}, input/source "
            f"sequence length: {input_sequence.shape[1]}, target sequence length: {target_sequence.shape[1]}."
        )
        src_mask = self._create_src_mask(input_sequence)
        trg_mask = self._create_trg_mask(target_sequence)
        encoder_output, encoder_attention_weights = self.encoder(input_sequence, src_mask)
        decoder_output, masked_mha_attention_weights, decoder_attention_weights = self.decoder(
            target_sequence, encoder_output, trg_mask, src_mask
        )
        logits = self.linear(decoder_output)
        return logits, encoder_attention_weights, masked_mha_attention_weights, decoder_attention_weights
