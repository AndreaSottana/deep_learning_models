from transformers import BertTokenizer, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from modules.bert_for_qa.preprocess_dataset import DatasetEncoder
from modules.bert_for_qa.fine_tuning import build_dataloaders, fine_tune_train_and_valid

if __name__ == '__main__':
    tokenizerr = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    import json
    with open("../data/SQuAD/train-v1.1.json", "r") as f:
        train = json.load(f)
    tok_enc = DatasetEncoder.from_dict_of_paragraphs(tokenizerr, train)
    input_ids, token_type_ids, attention_masks, start_positions, end_positions, dropped_samples = \
        tok_enc.tokenize_and_encode(max_len=384, log_interval=1000)
    for i in [input_ids, token_type_ids, attention_masks, start_positions, end_positions]:
        print(i.shape)
    print(dropped_samples, " samples dropped.")

    train_dataloader, valid_dataloader = build_dataloaders(
        input_ids,
        token_type_ids,
        attention_masks,
        start_positions,
        end_positions,
        batch_size=(16, 16),
        train_ratio=0.9,
    )

    model = BertForQuestionAnswering.from_pretrained(
        "bert-base-cased",  # Use the 12-layer BERT model with pre-trained weights, with a cased vocab.
        output_attentions=False,
        output_hidden_states=False,
    )
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # defaults: lr=5e-5, eps=1e-8
    training_epochs = 3
    training_steps = training_epochs * len(train_dataloader)  # epochs * number of batches
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=training_steps)

    model, training_stats = fine_tune_train_and_valid(
        train_dataloader,
        valid_dataloader,
        model,
        optimizer,
        training_epochs=training_epochs,
        lr_scheduler=None,
        save_model_path="trained_model.pt"
    )
