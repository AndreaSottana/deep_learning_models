from transformers import BertTokenizer
from modules.bert_for_qa.preprocess_dataset import DatasetEncoder

if __name__ == '__main__':
    tokenizerr = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    import json
    with open("../data/SQuAD/dev-v1.1.json", "r") as f:
        dev = json.load(f)
    tok_enc = DatasetEncoder.from_dict_of_paragraphs(tokenizerr, dev)
    input_ids, token_type_ids, attention_masks, start_positions, end_positions, dropped_samples = \
        tok_enc.tokenize_and_encode(100)
    for i in [input_ids, token_type_ids, attention_masks, start_positions, end_positions]:
        print(i.shape)
    print(dropped_samples, " samples dropped.")