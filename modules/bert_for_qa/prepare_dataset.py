import torch
from typing import Dict, List
from transformers import BertTokenizer
from .utils import format_time


def create_training_samples(input_dataset: Dict) -> List[Dict]:
    """
    This function takes as input the training, validation or test SQuAD dataset in its original format and converts it
    into a format which is more readily usable for our fine-tuning.
    :param input_dataset: a dictionary with two keys: "data" and "version". The first value is a list where each element
           corresponds to a paragraph and all its related questions and answers.
    :return: training_samples: a list where each element is a dictionary with 6 items, including a question, a context
             and the answer. The context is the reference text in which the answer can be found.
    """
    training_samples = []
    for article in input_dataset['data']:
        for paragraph in article['paragraphs']:
            for qas in paragraph['qas']:  # each paragraph has multiple questions and answers associated
                sample_dict = {
                    'answer_text': qas['answers'][0]['text'],
                    'context_text': paragraph['context'],
                    'qas_id': qas['id'],
                    'question_text': qas['question'],
                    'start_position_character': qas['answers'][0]['answer_starts'],
                    'title': article['title']
                }
                training_samples.append(sample_dict)
    return training_samples


def tokenize(training_samples: List[Dict], max_len: int):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    dropped_samples = 0  # number of samples dropped due to truncation (if answer is in truncated text we ignore sample)
    all_encoded_dicts = []
    all_question_start_indices = []
    all_question_end_indices = []

    for i, sample in enumerate(training_samples):
        answer_tokens = tokenizer.tokenize(sample['answer_text'])
        answer_replacement = " ".join(["[MASK]"] * len(answer_tokens))
        start_position_character = sample['start_position_character']
        end_position_character = sample['start_position_character'] + len(sample['answer_text'])
        context_with_replacement = sample['context_text'][:start_position_character] + answer_replacement + \
            sample['context_text'][end_position_character:]
        encoded_dict = tokenizer.encode_plus(
            sample['question_text'],
            context_with_replacement,
            add_special_tokens=True,  # Adds '[CLS]' and '[SEP]' tokens
            max_length=max_len,  # Pads or truncates sentences to `max_length`
            pad_to_max_length=True,
            # truncation = True,
            return_attention_mask=True,  # Construct attention masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        '''A dictionary containing the sequence pair and additional information. There are 3 keys, each value is a 
        torch.tensor of shape (1, max_len) and can be converted to just (max_len) by applying .squeeze():
        - 'input_ids': the ids of each token of the encoded sequence pair, with padding at the end
        - 'token_type_ids': 1 for token positions in  answer text, 0 elsewhere (i.e. in question and padding)
        - 'attention_mask': 1 for non "[PAD]" token, 0 for "[PAD]" tokens.'''
        is_mask_token = encoded_dict['input_ids'].squeeze() == tokenizer.mask_token_id
        mask_token_indices = is_mask_token.nonzero(as_tuple=False).squeeze()
        if len(mask_token_indices) != len(answer_tokens):
            dropped_samples += 1  # we drop sample due to answer being truncated
            continue
        question_start_index, question_end_index = mask_token_indices[0], mask_token_indices[-1]
        answer_token_ids = tokenizer.encode(
            sample['answer_text'],
            add_special_tokens=False,
            return_tensors='pt'
        )
        encoded_dict['input_ids'][0, question_start_index:question_end_index + 1] = answer_token_ids
        # Finally, replace the `[MASK]` tokens with the actual answer tokens
        all_encoded_dicts.append(encoded_dict)
        all_question_start_indices.append(question_start_index)
        all_question_end_indices.append(question_end_index)
    assert len(all_encoded_dicts) == len(training_samples) - dropped_samples, "Lengths check failed!"
    input_ids = torch.cat([encoded_dict['input_ids'] for encoded_dict in all_encoded_dicts], dim=0)
    token_type_ids = torch.cat([encoded_dict['token_type_ids'] for encoded_dict in all_encoded_dicts], dim=0)
    attention_masks = torch.cat([encoded_dict['attention_mask'] for encoded_dict in all_encoded_dicts], dim=0)
    start_positions = torch.cat(all_question_start_indices, dim=0)
    end_positions = torch.cat(all_question_end_indices, dim=0)
    return input_ids, token_type_ids, attention_masks, start_positions, end_positions, dropped_samples


