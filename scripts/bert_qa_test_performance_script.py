import yaml
import json
import torch
import logging.config
from transformers import BertTokenizer, BertForQuestionAnswering
from modules.bert_for_qa.preprocess_dataset import DatasetEncoder
from modules.bert_for_qa.prediction_loop import predict
from modules.bert_for_qa.scores import exact_match_rate, f1_score

if __name__ == '__main__':
    LOGGING_CONFIG = "modules/logging.yaml"
    with open(LOGGING_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    with open("data/SQuAD/dev-v1.1-small.json", "r") as f:
        dev = json.load(f)
    tok_enc = DatasetEncoder.from_dict_of_paragraphs(tokenizer, dev)
    input_ids, token_type_ids, attention_masks, start_positions, end_positions, dropped_samples = \
        tok_enc.tokenize_and_encode(
            max_len=384, log_interval=1000, start_end_positions_as_tensors=False, with_answers=True
        )
    for i in [input_ids, token_type_ids, attention_masks, start_positions, end_positions]:
        try:
            print(i.shape)
        except AttributeError:
            print(len(i))
    print(dropped_samples, " samples dropped.")
    model = BertForQuestionAnswering.from_pretrained("bert-base-cased")
    try:
        print(model.load_state_dict(torch.load("models/trained_model_epoch_3.pt")))
        logging.warning("Keeping model for predictions on CUDA device!")
    except RuntimeError:
        print(model.load_state_dict(torch.load("models/trained_model_epoch_3.pt", map_location=torch.device('cpu'))))
        logging.warning("Putting model for predictions on CPU as no GPU was found!")
    pred_start, pred_end = predict(input_ids, token_type_ids, attention_masks, model, batch_size=16)
    correct, total_indices, match_rate = exact_match_rate(start_positions, end_positions, pred_start, pred_end)
    all_f1, average_f1 = f1_score(start_positions, end_positions, pred_start, pred_end)
    test_set_results = {
        "correcr": correct,
        "total_indices": total_indices,
        "match_rate": match_rate,
        "all_f1": all_f1,
        "average_f1": average_f1
    }
    with open("models/test_set_results.json", "w") as f:
        json.dump(test_set_results, f)
