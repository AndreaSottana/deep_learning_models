import torch
import yaml
import logging.config
import modules.transformer.sublayers as sublayers
from modules.transformer.encoder import Encoder
from modules.transformer.decoder import Decoder
from modules.transformer.transformer import Transformer


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_param(self, k, v):
        setattr(self, k, v)


hp = Config(
    BATCH_SIZE=2,  # 100,  # 512,
    D_MODEL=16,  # THIS MUST BE EVEN OR ELSE IT WILL FAIL # 512
    MAX_SEQ_LEN=512,
    P_DROP=0.1,
    D_FF=20,  # 2048,
    HEADS=4,  # 8,
    LAYERS=3,  # 6,
    LR=1e-3,
    EPOCHS=40
)


if __name__ == '__main__':

    LOGGING_CONFIG = "../modules/logging.yaml"
    with open(LOGGING_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)

    run = 4

    if run == 1:

        ############## SIMPLE ENCODER - STEP BY STEP ##############################
        # toy_encodings = torch.rand(100, 768, 512, dtype=torch.float)
        # print(toy_encodings.shape)
        toy_vocab = torch.LongTensor([[1, 2, 3, 4, 0, 0]])
        toy_embedding_layer = sublayers.EmbeddingLayer(vocab_size=5, D=hp.D_MODEL)
        toy_embeddings = toy_embedding_layer(toy_vocab)
        print(toy_embeddings.shape)
        toy_PE_layer = sublayers.PositionalEncoding(D=hp.D_MODEL, max_seq_length=toy_embeddings.shape[1])
        toy_PEs = toy_PE_layer(toy_embeddings)
        toy_PEs22 = toy_PE_layer(toy_embeddings)
        toy_MHA_layer = sublayers.MultiHeadAttention(num_heads=hp.HEADS, D=hp.D_MODEL)
        # toy_MHA, toy_MHA_weights = toy_MHA_layer(toy_encodings, toy_encodings, toy_encodings)
        toy_MHA, toy_MHA_weights = toy_MHA_layer(toy_PEs, toy_PEs, toy_PEs)
        print(toy_MHA.shape, toy_MHA_weights.shape)
        # assuming we're adding residual connection (skipped as it's just addition between identical-shaped tensors)
        toy_Norm_layer = sublayers.LayerNormalization(D=hp.D_MODEL)
        toy_norm = toy_Norm_layer(toy_MHA)
        print(toy_norm.shape)
        toy_PWFFN_layer = sublayers.PositionWiseFeedForward(D=hp.D_MODEL, D_ff=hp.D_FF)
        toy_PWFFN = toy_PWFFN_layer(toy_norm)
        print(toy_PWFFN.shape)

    if run == 2:
        ############## SIMPLE ENCODER ##############################
        input_sequence = (10 * torch.rand(hp.BATCH_SIZE, 14)).long()
        mask_tensor = torch.rand(hp.BATCH_SIZE, 1, 14) > 0.5
        encoder = Encoder(
            src_vocab_size=10,
            D=hp.D_MODEL,
            num_heads=hp.HEADS,
            D_ff=hp.D_FF,
            num_layers=hp.LAYERS
        )
        encoder_output, encoder_attention_weights = encoder(input_sequence, mask_tensor)
        print(f"Encoder output shape: {encoder_output.shape},  expected: {hp.BATCH_SIZE, 14, hp.D_MODEL}")
        print(f"Encoder attention weights shape: {encoder_attention_weights.shape}, expected: { hp.BATCH_SIZE, hp.HEADS, 14, 14}")

    if run == 3:
        ############## SIMPLE DECODER ##############################
        target_sequence = (10 * torch.rand(hp.BATCH_SIZE, 27)).long()
        encoder_output = torch.rand(hp.BATCH_SIZE, 14, hp.D_MODEL)
        src_mask_tensor = torch.rand(hp.BATCH_SIZE, 1, 14) > 0.5  # 1 will be broadcaster to either 14 or 27
        trg_mask_tensor = torch.rand(hp.BATCH_SIZE, 27, 27) > 0.5
        decoder = Decoder(
            trg_vocab_size=10,
            D=hp.D_MODEL,
            num_heads=hp.HEADS,
            D_ff=hp.D_FF,
            num_layers=hp.LAYERS
        )
        decoder_output, masked_mha_attention_weights, decoder_attention_weights = decoder(
            target_sequence, encoder_output, trg_mask_tensor, src_mask_tensor
        )
        print(f"Decoder output shape: {decoder_output.shape}, expected: {hp.BATCH_SIZE, 27, hp.D_MODEL}.")
        print(f"Masked MHA weights shape: {masked_mha_attention_weights.shape}, expected: {hp.BATCH_SIZE, hp.HEADS, 27, 27}.")
        print(f"Decoder attention weights shape: {decoder_attention_weights.shape}, expected: {hp.BATCH_SIZE, hp.HEADS, 27, 14}.")

    if run == 4:
        ############## SIMPLE TRANSFORMER  ##############################
        input_sequence = (10 * torch.rand(hp.BATCH_SIZE, 14)).long()
        target_sequence = (10 * torch.rand(hp.BATCH_SIZE, 27)).long()
        transformer = Transformer(
            src_vocab_size=10,
            trg_vocab_size=13,
            D=hp.D_MODEL,
            num_heads=hp.HEADS,
            D_ff=hp.D_FF,
            max_seq_length=hp.MAX_SEQ_LEN,
            num_encoder_layers=hp.LAYERS,
            num_decoder_layers=hp.LAYERS
        )
        logits, encoder_attention_weights, masked_mha_attention_weights, decoder_attention_weights = transformer(
            input_sequence, target_sequence
        )
        print(
            logits.shape, " expected ", (hp.BATCH_SIZE, hp.MAX_SEQ_LEN, hp.D_MODEL), "\n",
            encoder_attention_weights.shape, " expected ", (hp.BATCH_SIZE, hp.HEADS, 14, 14), "\n",
            masked_mha_attention_weights.shape, " expected ", (hp.BATCH_SIZE, hp.HEADS, 27, 27), "\n",
            decoder_attention_weights.shape, " expected ", (hp.BATCH_SIZE, hp.HEADS, 27, 14)
        )
