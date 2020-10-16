import torch.nn as nn
from typing import Union, List
from transformers import PreTrainedTokenizerBase
from .prediction_loop import predict
from .preprocess_dataset import DatasetEncoder


class ChatBot:
    def __init__(self, context: str, tokenizer: PreTrainedTokenizerBase, model: nn.Module, max_len: int = 500) -> None:
        self.context = context
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len

    def answer(self, questions: Union[str, List[str]], disable_progress_bar: bool = True) -> Union[str, List[str]]:
        if isinstance(questions, str):
            questions = [questions]  # convert to list if a single question is given as string
        encoder = DatasetEncoder(
            tokenizer=self.tokenizer,
            input_dataset=[{'context_text': self.context, 'question_text': question} for question in questions]
        )
        input_ids, token_type_ids, attention_masks = encoder.tokenize_and_encode(
            max_len=self.max_len, with_answer=False
        )
        pred_start, pred_end = predict(
            input_ids, token_type_ids, attention_masks, self.model, 1, disable_progress_bar=disable_progress_bar
        )
        predicted_answers = [
            self.tokenizer.decode(input_ids[0, pred_start_i:pred_end_i + 1])
            for pred_start_i, pred_end_i in zip(pred_start, pred_end)
        ]
        if len(predicted_answers) == 1:
            return predicted_answers[0]  # return answer as string instead of list if there is only one question
        return predicted_answers



