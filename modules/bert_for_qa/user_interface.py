import torch
from transformers import BertTokenizer
from modules.bert_for_qa.prediction_loop import predict
from modules.bert_for_qa.preprocess_dataset import DatasetEncoder


class ChatBot:
    def __init__(self, context: str, tokenizer, model):
        self.context = context
        self.tokenizer = tokenizer
        self.model = model

    def answer(self, question: str):
        encoder = DatasetEncoder(
            tokenizer=self.tokenizer,
            input_dataset=[{
                'context_text': self.context,
                'question_text': question
            }]
        )
        input_ids, token_type_ids, attention_masks = encoder.tokenize_and_encode(
            max_len=500, with_answer=False
        )
        pred_start, pred_end = predict(input_ids, token_type_ids, attention_masks, self.model, batch_size=1)
        predicted_answer = self.tokenizer.decode(input_ids[0, pred_start:pred_end+1])
        return predicted_answer


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    model = torch.load("../../models/trained_model_full.pt", map_location=torch.device('cpu'))
    context = """
Boris Johnson is a British politician, author, and former 
journalist who has served as Prime Minister of the United Kingdom and Leader of the Conservative Party since 2019. 
He was Foreign Secretary from 2016 to 2018 and Mayor of London from 2008 to 2016. Johnson was Member of 
Parliament (MP) for Henley from 2001 to 2008 and has been MP for Uxbridge and South Ruislip since 2015. 
Ideologically, he identifies as a one-nation conservative.
"""
    bot = ChatBot(context, tokenizer, model)
    answer = bot.answer("Who is Boris Johnson?")
    print("Answer is: ", answer)
