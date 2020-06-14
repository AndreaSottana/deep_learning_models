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
    D_MODEL=6,  # THIS MUST BE EVEN OR ELSE IT WILL FAIL # 512
    MAX_SEQ_LEN=8,  # 512,
    P_DROP=0.1,
    D_FF=20,  # 2048,
    HEADS=3,  # 8,
    LAYERS=2,  # 6,
    LR=1e-3,
    EPOCHS=40
)


LOGGING_CONFIG = "../modules/logging.yaml"
with open(LOGGING_CONFIG, 'r') as f:
    config = yaml.safe_load(f)
logging.config.dictConfig(config)


if __name__ == '__main__':
    run = 3  # change according to which model you want to run

    if run == 1:
        ############## SIMPLE ENCODER - STEP BY STEP ##############################
        # toy_encodings = torch.rand(100, 768, 512, dtype=torch.float)
        # print(toy_encodings.shape)
        toy_vocab = torch.LongTensor([[1, 2, 3, 4, 0, 0]])
        toy_embedding_layer = sublayers.EmbeddingLayer(vocab_size=5, D=hp.D_MODEL)
        toy_embeddings = toy_embedding_layer(toy_vocab)
        print(toy_embeddings.shape)
        toy_PE_layer = sublayers.PositionalEncoding(D=hp.D_MODEL, seq_length=toy_embeddings.shape[1])
        toy_PEs = toy_PE_layer(toy_embeddings)
        print(toy_PEs.shape)
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
        input_sequence = (10 * torch.rand(hp.BATCH_SIZE, hp.MAX_SEQ_LEN)).long()
        mask_tensor = torch.rand(hp.BATCH_SIZE, hp.MAX_SEQ_LEN, hp.MAX_SEQ_LEN) > 0.5
        encoder = Encoder(
            src_vocab_size=10,
            seq_length=hp.MAX_SEQ_LEN,
            D=hp.D_MODEL,
            num_heads=hp.HEADS,
            D_ff=hp.D_FF,
            num_layers=hp.LAYERS
        )
        encoder_output, encoder_attention_weights = encoder(input_sequence, mask_tensor)
        print(encoder_output.shape, encoder_attention_weights.shape, sep="\n")

    if run == 3:
        ############## SIMPLE DECODER ##############################
        target_sequence = (10 * torch.rand(hp.BATCH_SIZE, hp.MAX_SEQ_LEN)).long()
        encoder_output = torch.rand(hp.BATCH_SIZE, hp.MAX_SEQ_LEN, hp.D_MODEL)
        src_mask_tensor = torch.rand(hp.BATCH_SIZE, hp.MAX_SEQ_LEN, hp.MAX_SEQ_LEN) > 0.5
        trg_mask_tensor = torch.rand(hp.BATCH_SIZE, hp.MAX_SEQ_LEN, hp.MAX_SEQ_LEN) > 0.5
        decoder = Decoder(
            trg_vocab_size=10,
            seq_length=hp.MAX_SEQ_LEN,  #NEED TO ADD AN ASSERT STATEMENT HERE
            D=hp.D_MODEL,
            num_heads=hp.HEADS,
            D_ff=hp.D_FF,
            num_layers=hp.LAYERS
        )
        decoder_output, masked_mha_attention_weights, decoder_attention_weights = decoder(
            target_sequence, encoder_output, trg_mask_tensor, src_mask_tensor
        )
        print(decoder_output.shape, masked_mha_attention_weights.shape, decoder_attention_weights.shape, sep="\n")

    if run == 4:
        ############## SIMPLE TRANSFORMER  ##############################
        input_sequence = (10 * torch.rand(hp.BATCH_SIZE, hp.MAX_SEQ_LEN)).long()
        target_sequence = (10 * torch.rand(hp.BATCH_SIZE, hp.MAX_SEQ_LEN)).long()
        transformer = Transformer(
            src_vocab_size=10,
            trg_vocab_size=12,
            seq_length=hp.MAX_SEQ_LEN,
            D=hp.D_MODEL,
            num_heads=hp.HEADS,
            D_ff=hp.D_FF,
            num_encoder_layers=hp.LAYERS,
            num_decoder_layers=hp.LAYERS
        )
        logits, encoder_attention_weights, masked_mha_attention_weights, decoder_attention_weights = transformer(
            input_sequence, target_sequence
        )
        print(
            logits.shape,
            encoder_attention_weights.shape,
            masked_mha_attention_weights.shape,
            decoder_attention_weights.shape,
            sep="\n"
        )
