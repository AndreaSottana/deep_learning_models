from typing import Dict,  List


def create_training_samples(input_dataset: Dict) -> List[Dict]:
    """
    This function takes as input the training, validation or test SQuAD dataset in its original format and converts it
    into a format which is more readily usable for our fine-tuning.
    :param input_dataset: a dictionary with two keys: "data" and "version". The first value is a list where each element
           corresponds to a paragraph and all its related questions and answers.
    :return: training_samples: a list where each element is a dictionary with 6 items, including a question, a context
             and the answer.
    """
    training_samples = []
    for article in input_dataset['data']:
        for paragraph in article['paragraphs']:
            for qas in paragraph['qas']:  # each paragraph has multiple questions and answers associated
                sample_dict = {
                    'answer_text': qas['answers']['text'],
                    'context_text': paragraph['context'],
                    'qas_id': qas['id'],
                    'question_text': qas['question'],
                    'start_position_character': qas['answers']['answer_starts'],
                    'title': article['title']
                }
                training_samples.append(sample_dict)
    return training_samples


