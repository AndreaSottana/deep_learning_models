import os
from torch.nn.modules import CrossEntropyLoss
from torch.optim.adam import Adam
from modules.transformer.dataset_build_and_train import *
from modules.transformer.transformer import Transformer


if __name__ == '__main__':
    DATA_DIR = '../data'  # os.getenv('DATA_DIR')


    class HyperParams:
        BATCH_SIZE = 3  # 100,  # 512,
        D_MODEL = 16  # THIS MUST BE EVEN OR ELSE IT WILL FAIL # 512
        # MAX_SRC_SEQ_LEN = 512,
        # MAX_TRG_SEQ_LEN = 512,
        P_DROP = 0.1
        D_FF = 20  # 2048,
        HEADS = 4  # 8,
        # LAYERS = 6,
        LR = 1e-3
        EPOCHS = 4  # 40


    hp = HyperParams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train, test, valid, src_field, trg_field = dataset_construction_from_raw_dataset(
        'de_core_news_sm',  # 'de',
        'en_core_web_sm',  # 'en',
        os.path.join(DATA_DIR, "machine_translation_toy_dataset"),
        train_filename='train',
        valid_filename='val',
        test_filename='test',
        filenames_exts=('.de', '.en'),
        min_freq=2,
        init_token='<sos>',
        eos_token='<eos>'
    )
    train_iter, valid_iter, test_iter = iterator_construction(
        train=train,
        valid=valid,
        test=test,
        batch_sizes=(hp.BATCH_SIZE, hp.BATCH_SIZE, hp.BATCH_SIZE),
        device=device
    )
    transformer_model = Transformer(
        len(src_field.vocab), len(trg_field.vocab), D=hp.D_MODEL, num_heads=hp.HEADS, D_ff=hp.D_FF, dropout=hp.P_DROP
    )
    criterion = CrossEntropyLoss(ignore_index=trg_field.vocab.stoi['<pad>'])
    # ignore_index specifies a target value that is ignored and does not contribute to the input gradient.
    training(
        transformer_model,
        epochs=hp.EPOCHS,
        train_iterator=train_iter,
        valid_iterator=valid_iter,
        optimizer=Adam(transformer_model.parameters(), lr=hp.LR, betas=(0.9, 0.98), eps=1e-9),  # NEED TO CHANGE
        loss_fn=criterion,
        device=device,
        log_interval=3,
        save_model=False,
        model_path=None  # os.path.join(os.getenv("MODELS_DIR"), "toy_transformer.pt")
    )
