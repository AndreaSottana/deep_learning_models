import logging
import time
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset
from tqdm import tqdm


logger = logging.getLogger(__name__)


def dataset_construction_from_raw_dataset(
        src_language: str,
        trg_language: str,
        path: str,
        filenames_exts: Tuple[str, str],
        min_freq: int = 1,
        train_filename: str = 'train',
        valid_filename: str = 'val',
        test_filename: str = 'test',
        init_token: Optional[str] = '<sos>',
        eos_token: Optional[str] = '<eos>',
) -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset, Field, Field]:
    """
    This function construct the train, validation and test datasets starting from raw files. It also builds the
    vocabulary from the training dataset. Raw files should be text files where each line correspond to a sentence
    in the respective language, and the extension should be language dependent. For example, if you have an English
    and German dataset, the train file should be called 'train.en' and 'train.de' respectively.
    :param src_language: the language of the source sequences, to be passed onto the Field tokenizer_language argument.
           Follows spacy's language abbreviations, i.e. 'en' for English, 'de' for German etc.
           See https://spacy.io/usage/models#languages for supporterd languages and their abbreviations.
    :param trg_language: the language of the target sequences, to be passed onto the Field tokenizer_language argument.
           Same conventions as for src_language (see above).
    :param path: the folder where the raw files are stored.
    :param filenames_exts: a tuple containing the extension to path for source and target language respectively.
           For German (source) and English (target), this would be filenames_exts = ('.de', '.en')
    :param min_freq: the minimum frequency a word must have, in the training corpus, in order to be included in
           the vocabulary. Default: 1.
    :param train_filename: the prefix of the train dataset (without extension). Default: 'train'.
    :param valid_filename: the prefix of the validation dataset (without extension). Default: 'val'.
    :param test_filename: the prefix of the test dataset (without extension). Default: 'test'.
    :param init_token: a token that will be prepended to every sentence, or None for no initial token. Default: '<sos>'.
    :param eos_token: a token that will be appended to every sentence, or None for no end-of-sentence token.
           Default: '<eos>'.
    :return: train: the training dataset, converted to a torchtest.datasets.TranslationDataset
             valid: the validation dataset, converted to a torchtest.datasets.TranslationDataset
             test: the test dataset, converted to a torchtest.datasets.TranslationDataset
             src_field: the Field object for the source dataset. Defines a datatype together with instructions for
             converting to Tensor. This might be needed if we want to convert new text to integers or viceversa using
             the vocabulary built with our input training corpus.
             trg_field: the Field object for the target dataset. See src_field for a description.
    """
    src_field = Field(
        sequential=True,
        use_vocab=True,
        init_token=init_token,
        eos_token=eos_token,
        tokenize='spacy',
        tokenizer_language=src_language,
        batch_first=True,
        is_target=False,
    )
    trg_field = Field(
        sequential=True,
        use_vocab=True,
        init_token=init_token,
        eos_token=eos_token,
        tokenize='spacy',
        tokenizer_language=trg_language,
        batch_first=True,
        is_target=True
    )

    train, valid, test = TranslationDataset.splits(
        exts=filenames_exts,
        fields=(src_field, trg_field),
        path=path,
        train=train_filename,  # these will be suffixed with the extensions given in the exts tuple.
        validation=valid_filename,
        test=test_filename
    )

    src_field.build_vocab(train, min_freq=min_freq)
    trg_field.build_vocab(train, min_freq=min_freq)

    return train, valid, test, src_field, trg_field


def iterator_construction(
        train: TranslationDataset,
        valid: TranslationDataset,
        test: TranslationDataset,
        batch_sizes: Tuple[int, int, int],
        device: Union[str, torch.device]
) -> Tuple[BucketIterator, BucketIterator, BucketIterator]:
    """
    This function takes torchtext.data.TranslationDataset's as input and outputs the correspondent BucketIterator's,
    splitting the datasets into batches. This iterator batches examples of similar lengths together, minimizing the
    amount of padding needed while producing freshly shuffled batches for each new training epoch.
    :param train: a torchtest.data.TranslationDataset representing the training dataset.
    :param valid: a torchtest.data.TranslationDataset representing the validation dataset.
    :param test: a torchtest.data.TranslationDataset representing the test dataset.
    :param batch_sizes: a tuple of 3 integers, each representing the batch size for the train, validation and test set
           respectively.
    :param device: the device to be used for the calculations. Can be a str (e.g. 'cuda') or torch.device object.
    :return: train_iter: a torchtext.data.BucketIterator, the iterator for the training dataset.
             valid_iter: a torchtext.data.BucketIterator, the iterator for the validation dataset.
             test_iter: a torchtext.data.BucketIterator, the iterator for the test dataset.
    """
    train_iter, valid_iter, test_iter = BucketIterator.splits(
        datasets=(train, valid, test),
        batch_sizes=batch_sizes,
        sort=True,
        device=device
    )
    return train_iter, valid_iter, test_iter


def training(
        model: nn.Module,
        epochs: int,
        train_iterator: BucketIterator,
        valid_iterator: BucketIterator,
        optimizer: Any,  # any optimizer from the module torch.optim
        loss_fn: Any,  # any loss from the module torch.nn
        device: Union[str, torch.device],
        log_interval: int = 10,
        save_model: bool = False,
        model_path: Optional[str] = None,
) -> None:
    """
    This function implements the training of a model subclassed from the torch.nn.Module class. It runs the model both
    in training and evaluation mode, and logs the losses. While any suitable model can be inputted, this function has
    been specifically designed to train the Transformer model, as described in the paper 'Attention is All You Need',
    Dec 2017, https://arxiv.org/pdf/1706.03762.pdf
    :param model: the instantiated model to be trained. It should be an instance of torch.nn.Module
    :param epochs: the number of training epochs. Each epoch will have different, freshly shuffled batches.
    :param train_iterator: the train iterator encoded as torchtext.data.BucketIterator
    :param valid_iterator: the validation iterator encoded as torchtext.data.BucketIterator
    :param optimizer: an appropriate optimizer from the module torch.optim
    :param loss_fn: an appropriate loss function from the module torch.nn
           For the Transformer, this should be CrossEntropyLoss
    :param device: the device to be used for the calculations. Can be a str (e.g. 'cuda') or torch.device object.
    :param log_interval: the interval, within each training epoch, for logging the losses. Default: 10, i.e. the losses
           will be logged every 10 batches. If there are less than 10 batches, the losses will only be logged at the
           end of each epoch by default.
    :param save_model: whether to save the model to the disk. Default: False.
    :param model_path: the path (including the extensions) where the model is saved. Suggested extensions: '.pt'.
           Default: None. If save_model is set to True, model_path must be explicitly specified by the user.
    :return: None.
    """
    global_time = time.time()
    for epoch in tqdm(range(1, epochs + 1)):
        epoch_start_time = time.time()
        logger.info(f"Starting training mode for epoch {epoch}.")
        model.train()
        cumulative_train_loss = 0.
        for batch_num, iterator in enumerate(train_iterator):
            batch_start_time = time.time()
            input_sequence = iterator.src
            target_sequence = iterator.trg
            optimizer.zero_grad()

            predicted_sequence, _, _, _ = model.to(device)(input_sequence, target_sequence)
            # predicted_sequence dim: (batch_size, trg_seq_length, trg_vocab_size)
            train_loss = loss_fn(predicted_sequence.permute(0, 2, 1), target_sequence)
            # CrossEntropyLoss takes as input a tensor of shape (batch_size,  trg_vocab_size, trg_sequence_length) and
            # as target a tensor of shape (batch_size, trg_sequence_length)
            cumulative_train_loss += train_loss.item()  # * input_sequence.shape[0]  # multiplying loss by batch_size
            train_loss.backward()
            optimizer.step()
            if (batch_num + 1) % log_interval == 0:
                logger.info(f"Starting validation mode for epoch {epoch}. Validating a random batch.")
                model.eval()
                with torch.no_grad():
                    input_sequence = next(iter(valid_iterator)).src.detach()  # picking first batch for evaluation
                    target_sequence = next(iter(valid_iterator)).trg.detach()
                    predicted_sequence, _, _, _ = model.to(device)(input_sequence, target_sequence)
                    valid_loss = loss_fn(predicted_sequence.permute(0, 2, 1), target_sequence)
                logger.warning(
                    f"Epoch {epoch}. {batch_num + 1}/{len(train_iterator)} batches. Training loss: "
                    f"{cumulative_train_loss:.5}. Validation loss: {valid_loss:.5}."
                    f"\nAverage speed: {1000 * (time.time() - epoch_start_time) / (batch_num + 1):.5} ms/batch. "
                    f"Current batch processing time: {1000 * (time.time() - batch_start_time):.5} ms."
                )

        logger.warning(
            f"END OF EPOCH {epoch}. Average speed: {(time.time() - global_time) / epoch:.5}s/epoch. Current epoch "
            f"processing time: {time.time() - epoch_start_time:.5}s."
        )

    if save_model:
        assert model_path is not None, 'When save_model is True, model_path argument must be explicitly specified.'
        torch.save(model, model_path)
        logger.info(f"Model saved as {model_path}.")
