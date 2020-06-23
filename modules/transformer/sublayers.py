import torch
import torch.nn as nn
import math as m
import logging
from typing import Tuple, Optional


logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    This class implements the multi-head self attention mechanism of the first sublayer of the encoder layer and the
    second sublayer of the decoder. It can also be used to implement the masked multi-head attention (first sublayer
    of the decoder) by specifying a mask tensor when calling the forward method. The default mask is None.
    """
    def __init__(self, num_heads: int, D: int, device: str = 'cpu', dropout: float = 0.1) -> None:
        """
        :param num_heads: the number of attention heads. The paper uses 8 heads.
        :param D: the length of the input matrix (per token). For the first encoder layer, this is the embedding
               matrix. In the paper D = 512.
        :param device: the device on which to run the computations. Default: 'cpu'.
        :param dropout: the dropout applied to the multi-head attention matrix. Default: 0.1
        """
        assert D % 2 == 0, f"D must be an even number, got {D}. Please reset it accordingly."
        assert D % num_heads == 0, \
            f"D must be an exact multiple of num_heads. Got {D} and {num_heads}. Please change your values accordingly."
        if num_heads % 2 == 1:
            logger.warning(f"num_heads should be an even number, got {num_heads}. Please consider resetting.")
        super().__init__()
        self.D = D
        self.num_heads = num_heads
        self.d_k = self.D // self.num_heads  # d_k is an integer. The paper sets it at 64, which is 512/8

        self.dropout = nn.Dropout(dropout)
        self.Q_linears = nn.ModuleList([nn.Linear(self.D, self.d_k).to(device) for _ in range(self.num_heads)])
        self.K_linears = nn.ModuleList([nn.Linear(self.D, self.d_k).to(device) for _ in range(self.num_heads)])
        self.V_linears = nn.ModuleList([nn.Linear(self.D, self.d_k).to(device) for _ in range(self.num_heads)])
        # .to(device) required as modules within a list comprehension are not always visible directly in the
        # computational graph. Elsewhere it is not needed, as doing transformer = Transformer(...).to(device) at call
        # point will automatically apply .to(device) to any tensor in all modules used in the Transformer.

        self.softmax = nn.Softmax(-1)
        self.mha_linear = nn.Linear(self.D, self.D)

    def _attention_head(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the "Scaled Dot-Product Attention" as described in the paper. We compute the dot product of the
        query with all keys, divide each by d_k, and apply a softmax activation to obtain the weights on the values.
        Computations are parallelised using matrix multiplications. The division by d_k is just a practical scaling
        factor which leads to stabler gradients. This is particularly useful for large values of dk, where the dot
        products grow large in magnitude, pushing the softmax function into regions where it has extremely small
        gradients. To make computations efficient, each matrix has been turned into a rank-3 tensor by adding the batch
        as the first dimension.
        :param query: query matrix. Dim: (batch_size, sequence_length, d_k)
        :param key: key matrix. Dim: (batch_size, sequence_length, d_k)
        :param value: value matrix. Dim: (batch_size, sequence_length, d_k)
        :param mask: the mask tensor to use when calculating masked multi-head attention (i.e. in the decoder). This
               must be a tensor of booleans, shape (batch_size, output_sequence_length, output_sequence_length). If
               provided, the values of the masked tensor when the mask is True will be replaced with -1e9 (the paper
               uses -inf). The mask tensor and the tensor to mask (attention_scores) must have the same shape i.e.
               (batch_size, sequence_length, sequence_length)
        :return: attention weights: the weights. Dim: (batch_size, sequence_length, sequence_length)
                 attention: the attention matrix. Dim: (batch_size, sequence_length, d_k)
        """
        attention_scores = torch.matmul(query, key.permute(0, 2, 1)) / m.sqrt(self.d_k)
        if mask is not None:
            assert attention_scores.shape[0] == mask.shape[0] and attention_scores.shape[2] == mask.shape[2], \
                f'The mask tensor must have shape {attention_scores.shape} or torch.Size([{attention_scores.shape[0]}' \
                f', 1, {attention_scores.shape[2]}]). Got {mask.shape} instead.'
            logger.info(f"Using a mask tensor of shape {mask.shape}.")
            attention_scores = attention_scores.masked_fill(mask, -1e9)
        else:
            logger.info("Mask tensor not provided.")
        attention_weights = self.softmax(attention_scores)  # (batch_size, sequence_length, sequence_lenght)
        attention = torch.matmul(attention_weights, value)
        return attention, attention_weights

    def forward(
            self,
            pre_query: torch.Tensor,
            pre_key: torch.Tensor,
            pre_value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the forward pass for the multi-head (self) attention layer.
        Takes the learned query, key and value matrices from the previous encoder/decoder and feeds them into the
        current encoder/decoder, computing the attention weights and the attention per each head.
        As the paper notes, instead of performing a single attention function, it is  beneficial to linearly project
        the queries, keys and values 'num_heads' times with different, learned linear projections. Multi-head attention
        allows the model to jointly attend to information from different representation subspaces at different
        positions. All the heads are then concatenated and once again projected, resulting in the final values.
        :param pre_query: output query from the previous encoder. Dim: (batch_size, sequence_length, D)
        :param pre_key: output key from the previous encoder. Dim: (batch_size, sequence_length, D)
        :param pre_value: output value from the previous encoder. Dim: (batch_size, sequence_length, D)
        :param mask: the mask tensor to use when calculating masked multi-head attention (i.e. in the decoder). This
               must be a tensor of booleans, shape (batch_size, output_sequence_length, output_sequence_length).
               Default: None
        :return: multi_head_attention. Dim: (batch_size, sequence_length, D) where D is the number of attention
                 heads num_heads multiplied by d_k.
                 attention_weights: this has now become a rank-4 tensor, as the weights from multiple heads have been
                 stacked and not concatenated. Dim: (batch_size, num_heads, sequence_length, sequence_length).
                 The weights are returned just so they can be plotted if one wishes. They are not used in any future
                 calculation outside this module.
        """
        assert all([pre_query.shape[i] == pre_key.shape[i] == pre_value.shape[i] for i in [0, 2]]), \
            f"query, key and value must have the same first and third dimension (bath size and D respectively). " \
            f"Got shapes {pre_query.shape}, {pre_key.shape}, and {pre_value.shape} instead."
        assert len(pre_query.shape) == 3, \
            f"Expected rank-3 tensor, including batch size, got rank-{len(pre_query.shape)} instead. " \
            f"If using single-batch, consider adding '.unsqueeze(0)' to your input tensor."
        assert pre_query.shape[2] == self.D, \
            f"The last dimension of the input tensor must be equal to D={self.D}. Got {pre_query.shape[2]} instead."
        logger.info(
            f"Implementing the forward pass for MultiHeadAttention. Batch size: {pre_query.shape[0]}, sequence length: "
            f"{pre_query.shape[1]}, embedding/inner layer dimension: {pre_query.shape[2]}."
        )
        Q = [linear(pre_query) for linear in self.Q_linears]  # (batch_size, sequence_length_, d_k)
        K = [linear(pre_key) for linear in self.K_linears]
        V = [linear(pre_value) for linear in self.V_linears]

        output_per_head = []  # list of num_heads tensors, each with dim: (batch_size, sequence_length, d_k)
        attention_weights_per_head = []
        # list of num_heads tensors, each with dim: (batch_size, sequence_length, sequence_length)

        for q, k, v in zip(Q, K, V):
            attention, attention_weights = self._attention_head(q, k, v, mask)
            attention_weights_per_head.append(attention_weights)
            output_per_head.append(attention)

        multi_head_attention = torch.cat(output_per_head, dim=-1)
        attention_weights = torch.stack(attention_weights_per_head, dim=1)  # dim=1 so we keep batch_size first
        multi_head_attention = self.mha_linear(self.dropout(multi_head_attention))
        return multi_head_attention, attention_weights


class LayerNormalization(nn.Module):
    """
    This class implements the layer normalization as described in the paper. After each multi-head (self) attention
    module and after each position-wise feed forward module, the paper implements a residual connection, followed by
    layer normalization. This module adds a dropout as standard.
    """
    def __init__(self, D: int, dropout: float = 0.1) -> None:
        """
        :param D: the length of the input matrix (per token). For the first encoder layer, this is the embedding
               matrix. In the paper D = 512.
        :param dropout: the dropout applied to the normalized layer matrix. Default: 0.1
        """
        assert D % 2 == 0, f"D must be an even number, got {D}. Please reset it accordingly."
        super().__init__()
        self.D = D
        self.layernorm = nn.LayerNorm(D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the layer normalization.
        :param Z: the output from the previous layer (either the multi-head attention or position-wise feed forward)
               Dim: (batch_size, sequence_length, D)
        :return: normalized_layer: the output, after applying layer normalization.mDim: (batch_size, sequence_length, D)
        """
        assert len(Z.shape) == 3, \
            f"Expected rank-3 tensor, including batch size, got rank-{len(Z.shape)} instead. " \
            f"If using single-batch, consider adding '.unsqueeze(0)' to your input tensor."
        assert Z.shape[2] == self.D, \
            f"The last dimension of the input tensor must be equal to D={self.D}. Got {Z.shape[2]} instead."
        logger.info(
            f"Implementing the forward pass for LayerNormalization. Batch size: {Z.shape[0]}, sequence length: "
            f"{Z.shape[1]}, embedding/inner layer dimension: {Z.shape[2]}."
        )
        normalized_layer = self.layernorm(Z)
        normalized_layer = self.dropout(normalized_layer)
        return normalized_layer


class PositionWiseFeedForward(nn.Module):
    """
    This class implements the position-wise feed forward network. As described in the paper, in addition to attention
    sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which
    is applied to each position separately and identically. This consists of two linear transformations with a ReLU
    activation in between. This means that they run a feed-forward network over a rank 3 tensor
    (including batch size), over the sequence dimension.
    In the paper, the dimensionality of input and output is D = 512, and the inner-layer has dimensionality D_ff = 2048.
    It is unclear why the inputs are projected to a larger D_ff dimension before being projected back down to D
    dimensions, but some sources suggests that it is a convergence trick which enables re-scaling of the feature
    vectors independently with each other. See https://graphdeeplearning.github.io/post/transformers-are-gnns/
    """
    def __init__(self, D: int, D_ff: int, dropout: float = 0.1) -> None:
        """
        :param D: the length of the input matrix (per token). For the first encoder layer, this is the embedding
               matrix. In the paper D = 512.
        :param D_ff: the output dimension of the first layer and input dimension of the second (final) layer.
        :param dropout: the dropout applied to the second layer. Default: 0.1.
        """
        assert D % 2 == 0, f"D must be an even number, got {D}. Please reset it accordingly."
        super().__init__()
        self.D = D
        self.linear_1 = nn.Linear(D, D_ff)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(D_ff, D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the position-wise feed forward layer.
        :param Z: The output from the multi-head attention layer, after applying residual connection and layer
               normalization. Dim: (batch_size, sequence_length, D)
        :return: pwff_output: The output after applying the position-wise feed forward module.
                 Dim: (batch_size, sequence_length, D)
        """
        assert len(Z.shape) == 3, \
            f"Expected rank-3 tensor, including batch size, got rank-{len(Z.shape)} instead. " \
            f"If using single-batch, consider adding '.unsqueeze(0)' to your input tensor."
        assert Z.shape[2] == self.D, \
            f"The last dimension of the input tensor must be equal to D={self.D}. Got {Z.shape[2]} instead."
        logger.info(
            f"Implementing the forward pass for PositionWiseFeedForward. Batch size: {Z.shape[0]}, sequence length: "
            f"{Z.shape[1]}, embedding/inner layer dimension: {Z.shape[2]}."
        )
        pwff = self.relu(self.linear_1(Z))
        pwff_output = self.linear_2(self.dropout(pwff))
        return pwff_output


class EmbeddingLayer(nn.Module):
    """
    This class implements the embedding layer. This is the first layer of the encoder and the decoder. It takes as
    input the input (encoder) or target (decoder) sequence and converts it into an embeddings tensor.
    """
    def __init__(self, vocab_size: int, D: int, pad_idx: int = 0) -> None:
        """
        :param vocab_size: the length of the vocabulary, i.e. the size of the dictionary of embeddings.
        :param D: the length of the embedding matrix. In the paper D = 512.
        :param pad_idx: The index in the vocabulary for the '<pad>' token. Default: 0.
        """
        assert D % 2 == 0, f"D must be an even number, got {D}. Please reset it accordingly."
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.emb = nn.Embedding(vocab_size, D, padding_idx=pad_idx)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass for the embeddings layer.
        :param input_sequence: The input sentences, each sentence represented by numbers corresponding to the index
               of the sentence's words in the vocabulary. This must be a LongTensor. Dim: (batch_size, sequence_length)
        :return: embeddings: The embeddings tensor. Dim: (batch_size, sequence_length, D)
        """
        assert len(input_sequence.shape) == 2, \
            f"Expected rank-2 tensor, including batch size, got rank-{len(input_sequence)} instead." \
            f"If using single-batch, consider adding '.unsqueeze(0)' to your input sequence."
        logger.info(
            f"Implementing the forward pass for EmbeddingLayer. Size of the vocabulary: {self.vocab_size}. Batch size: "
            f"{input_sequence.shape[0]}, sequence length: {input_sequence.shape[1]}, embedding dimension: {self.D}."
        )
        embeddings = self.emb(input_sequence)
        embeddings = embeddings * m.sqrt(self.D)  # WHY ARE WE TAKING SQRT(D)?
        return embeddings


class PositionalEncoding(nn.Module):
    """
    This class implements the positional encoding. As noted in the paper, since our model contains no recurrence and no
    convolution, in order for the model to make use of the order of the sequence, we must inject some information about
    the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the
    input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension
    as the embeddings, so that the two can be summed. This class implements the formula as described in the paper.
    """
    def __init__(self, D: int, max_seq_length: int = 512, dropout: float = 0.1) -> None:
        """
        :param D: the length of the input matrix (per token). For the first encoder layer, this is the embedding
               matrix. In the paper D = 512.
        :param max_seq_length: The maximum length of a sentence which can be used in this model. If a sentence is longer
               it must be truncated (this module does not provide a feature for truncation, it's up to the user to
               ensure sentences are within this limit). Most sentences will be much shorter. Default: 512.
        :param dropout: the dropout applied to the positionally encoded embedding tensor.
        """
        super().__init__()
        assert D % 2 == 0, f"D must be an even number, got {D}. Please reset it accordingly."
        self.D = D
        self.emb = EmbeddingLayer(vocab_size=1, pad_idx=0, D=D)
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(0, max_seq_length).unsqueeze(1).float()  # dim: (max_seq_length, 1)
        pos_enc = torch.zeros(max_seq_length, D)  # initializing the positional encoding matrix, dim: (max_seq_len, D)
        even_index = torch.arange(0, D, step=2).float()  # dim: (D/2)
        div_term = torch.pow(10000, (even_index / torch.Tensor([D]))).float()
        pos_enc[:, 0::2] = torch.sin(pos / div_term)
        pos_enc[:, 1::2] = torch.cos(pos / div_term)
        pos_enc = pos_enc.unsqueeze(0)
        # Due to broadcasting, torch.Size([seq_length, 1]) / torch.Size([D/2]) gives torch.Size([seq_length, D/2])
        self.register_buffer("pos_enc", pos_enc)  # stores it as untrainable parameter (NO grad, learning, optimization)

    def forward(self, embs: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the positional encoding layer.
        :param embs: The embedding tensor. Dim: (batch_size, sequence_length, D)
        :return: pos_enc_emb: The positionally encoded embedding tensor. Dim: (batch_size, sequence_length, D)
        """
        assert self.pos_enc.shape[1] >= embs.shape[1], \
            f"The lenght of the input sequence {embs.shape[1]} cannot be bigger than the maximum sequence length " \
            f"allowed, i.e. {self.pos_enc.shape[1]}. Consider truncating your input sequence or splitting it up."
        assert len(embs.shape) == 3, \
            f"Expected rank-3 tensor, including batch size, got rank-{len(embs.shape)} instead. " \
            f"If using single-batch, consider adding '.unsqueeze(0)' to your input tensor."
        assert embs.shape[2] == self.D, \
            f"The last dimension of the embeddings tensor must be equal to D={self.D}. Got {embs.shape[2]} instead."
        logger.info(
            f"Implementing the forward pass for PositionalEncoding. Batch size: {embs.shape[0]}, sequence length: "
            f"{embs.shape[1]}, embedding dimension: {embs.shape[2]}."
        )
        pos_enc = self.pos_enc[:, :embs.shape[1]].detach()
        # detached because we don't want to learn weights (it's fixed, not a parameter). Dim: (1, sequence_length, D)
        # :embs.shape[1] ensures only values up to the effective sequence_length are retained (max_sequence length can
        # be much bigger than real sequence_length.
        pos_enc_emb = self.dropout(embs + pos_enc)
        return pos_enc_emb
