import torch
from typing import Dict, List, Tuple, Optional
from transformers import PreTrainedTokenizerBase
import logging
from time import time
from .utils import format_time


logger = logging.getLogger(__name__)


class DatasetEncoder:
    """
    This class handles all the preprocessing steps to convert the raw labelled dataset (consisting, at a minimum, of
    context-question-answer triplets plus other optional metadata) to the tensor inputs into the BERT model for
    fine-tuning on question answering tasks.
    The class can be initialised using a ready-made input dataset, or via the from_dict_of_paragraphs classmethod
    using a SQuAD-like dictionary dataset.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, input_dataset: List[Dict]):
        """
        :param tokenizer: the tokenizer used to tokenize the text. Must be a class derived from PreTrainedTokenizerBase.
        :param input_dataset: a list where each element is a dictionary with 4 to 6 items, namely
               - 'answer_text': a str with the answer
               - 'context_text': a str with the full reference text in which the answer can be found,
               - 'qas_id': optional, a hash-like str which is unique to each question-answer pair,
               - 'question_text': a str with the question text,
               - 'start_position_character': an int corresponding to the index of the first character of the answer in
                  the context text,
               - 'title': optional (for SQuAD only): a str with the title of the article where the context is taken
                 from. This is not used directly and is for reference purposes only.
        """
        self._tokenizer = tokenizer
        self._input_dataset = input_dataset

    def __len__(self):
        return len(self._input_dataset)

    def __getitem__(self, item):
        return self._input_dataset[item]

    @classmethod
    def from_dict_of_paragraphs(cls, tokenizer: PreTrainedTokenizerBase, input_dataset: Dict):
        """
        A classmethod to instantiate the class from a SQuAD-like dictionary dataset.
        :param tokenizer: the tokenizer used to tokenize the text. Must be a class derived from PreTrainedTokenizerBase.
        :param input_dataset: passed as argument of _create_training_samples_from_dict_of_paragraphs
        :return: an instance of DatasetEncoder ready for use.
        """
        training_samples = cls(tokenizer, cls._create_training_samples_from_dict_of_paragraphs(input_dataset))
        return training_samples

    @staticmethod
    def _create_training_samples_from_dict_of_paragraphs(input_dict: Dict) -> List[Dict]:
        """
        This is called by the from_dict_of_paragraphs class method when instantiating the class with a SQuAD-like
        dataset, and it converts the dataset into a format which is more readily usable for our fine-tuning.
        :param input_dict: a dictionary with two keys: "data" and "version". The first value is a list where each
               element corresponds to a paragraph and all its related questions and answers.
        :return: training_samples: a list where each element is a dictionary with 5 or 6 items, including a question, a
                 context and the answer. The context is the reference text in which the answer can be found.
        """
        training_samples = []
        for article in input_dict['data']:
            for paragraph in article['paragraphs']:
                for qas in paragraph['qas']:  # each paragraph has multiple questions and answers associated
                    sample_dict = {
                        'answers': qas['answers'],
                        'context_text': paragraph['context'],
                        'qas_id': qas['id'],
                        'question_text': qas['question'],
                        'title': article['title']
                    }
                    training_samples.append(sample_dict)
        return training_samples

    def tokenize_and_encode(
            self, max_len: int, log_interval: Optional[int] = None
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, int]:
        """
        This method converts the input dataset into  a number of tensors ready to train the BERT model for question
        answering. It takes as input the maximum length to pad the text to. Any sample where the answer falls outside
        (or partially outside) the question + answer sentence pair after its truncation to max_len is dropped from the
        dataset. The remaining N samples are tokenized and encoded.
        :param max_len: an int; the maximum length to pad the question + answer sentence pair sequence to. Training
               time is quadratic with max_len, however if max_len is too low, more answers will fall outside the limit
               and will be truncated, making those samples unusable and therefore hurting accuracy due to the loss of
               information. GPU or CPU memory limits also need be taken into account when finding the best trade-off.
        :param log_interval: the interval when to log the encoding status. Default: None
        :return: input_ids: torch.tensor of shape (N, max_len) representing the ids of each token of the N encoded
                 sequence pairs, with padding at the end.
                 token_type_ids: torch.tensor of shape (N, max_len) where each Nth dimension is filled with 1 for token
                 positions in the context text, 0 elsewhere (i.e. in question and padding)
                 attention_masks: torch.tensor of shape (N, max_len) where each Nth dimension is filled with 1 for
                 non-"[PAD]" tokens, 0 for "[PAD]" tokens.
                 start_positions: torch.tensor of shape (N, m) containing the index of the first answer token for each
                 answer, where m is the number of possible correct answers given to the model (m should be 1 for
                 training but can be higher for testing).
                 end_positions: torch.tensor of shape (N, m) containing the index of the last answer token for each answer
                 dropped_samples: int, the number of samples dropped from the dataset due to the answer falling outside
                 (or partially outside) the question + answer sentence pair truncated to max_len. For N encoded
                 sequence pairs, dropped_samples = len(training_samples) - N
        """
        dropped_samples = 0
        all_encoded_dicts = []
        all_question_start_indices = []
        all_question_end_indices = []

        t_i = time()  # initial time
        for i, sample in enumerate(self._input_dataset):
            if log_interval is not None and i % log_interval == 0 and i != 0:
                logger.warning(
                    f"Encoding sample {i} of {len(self._input_dataset)}. Elapsed: {format_time(time() - t_i)}. "
                    f"Remaining: {format_time((time() - t_i) / i * (len(self._input_dataset) - i))}."
                )
            for possible_answer in sample['answers']:
                possible_starts = []
                possible_ends = []
                answer_tokens = self._tokenizer.tokenize(possible_answer['text'])
                answer_replacement = " ".join(["[MASK]"] * len(answer_tokens))
                start_position_character = possible_answer['answer_start']
                end_position_character = possible_answer['answer_start'] + len(possible_answer['text'])
                context_with_replacement = sample['context_text'][:start_position_character] + answer_replacement + \
                    sample['context_text'][end_position_character:]
                encoded_dict = self._tokenizer.encode_plus(
                    sample['question_text'],
                    context_with_replacement,
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]' tokens
                    max_length=max_len,
                    padding='max_length',  # Pad or truncates sentences to `max_length`
                    truncation=True,
                    return_attention_mask=True,  # Construct attention masks.
                    return_tensors='pt',  # Return pytorch tensors.
                )
                '''A dictionary containing the sequence pair and additional information. There are 3 keys, each value is a 
                torch.tensor of shape (1, max_len) and can be converted to just (max_len) by applying .squeeze():
                - 'input_ids': the ids of each token of the encoded sequence pair, with padding at the end
                - 'token_type_ids': 1 for token positions in  answer text, 0 elsewhere (i.e. in question and padding)
                - 'attention_mask': 1 for non "[PAD]" token, 0 for "[PAD]" tokens.'''
                is_mask_token = encoded_dict['input_ids'].squeeze() == self._tokenizer.mask_token_id
                mask_token_indices = is_mask_token.nonzero(as_tuple=False)
                if len(mask_token_indices) != len(answer_tokens):
                    dropped_samples += 1  # we drop sample due to answer being truncated
                    continue
                question_start_index, question_end_index = mask_token_indices[0], mask_token_indices[-1]
                possible_starts.append(question_start_index)
                possible_ends.append(question_end_index)
                answer_token_ids = self._tokenizer.encode(
                    possible_answer['text'],
                    add_special_tokens=False,
                    return_tensors='pt'
                )
            encoded_dict['input_ids'][0, question_start_index:question_end_index + 1] = answer_token_ids
            # Finally, replace the "[MASK]" tokens with the actual answer tokens
            all_encoded_dicts.append(encoded_dict)
            all_question_start_indices.append(possible_starts)
            all_question_end_indices.append(possible_ends)
        print(len(all_encoded_dicts), len(self._input_dataset) - dropped_samples, len(self._input_dataset), dropped_samples)
        print(all_question_end_indices[:10])
        assert len(all_encoded_dicts) == len(self._input_dataset) - dropped_samples, "Lengths check failed!"
        input_ids = torch.cat([encoded_dict['input_ids'] for encoded_dict in all_encoded_dicts], dim=0)
        token_type_ids = torch.cat([encoded_dict['token_type_ids'] for encoded_dict in all_encoded_dicts], dim=0)
        attention_masks = torch.cat([encoded_dict['attention_mask'] for encoded_dict in all_encoded_dicts], dim=0)
        start_positions = torch.tensor(all_question_start_indices)  # torch.cat(all_question_start_indices, dim=0)
        end_positions = torch.tensor(all_question_end_indices)  # torch.cat(all_question_end_indices, dim=0)
        return input_ids, token_type_ids, attention_masks, start_positions, end_positions, dropped_samples
