from transformers import BertForQuestionAnswering, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from .utils import set_hardware_acceleration, format_time
from typing import Optional
from tqdm import tqdm
from time import time
import torch
import logging


logger = logging.getLogger(__name__)


def fine_tune(
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        batch_size: int,
        training_epochs: int = 3,
        train_ratio: float = 0.9,
        save_model_path: Optional[str] = None,
        device_: Optional[str] = None  # if None, it automatically detects if a GPU is available, if not uses a CPU
):
    assert input_ids.shape == token_type_ids.shape == attention_masks.shape, "Some input shapes are wrong"
    assert input_ids.shape[0] == len(start_positions) == len(end_positions), "Some input shapes are wrong!"
    model = BertForQuestionAnswering.from_pretrained(
        "bert-base-cased",  # Use the 12-layer BERT model, with a cased vocab.
        output_attentions=False,
        output_hidden_states=False,
    )
    device = set_hardware_acceleration(default=device_)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # defaults: lr=5e-5, eps=1e-8
    dataset = TensorDataset(
        input_ids, token_type_ids, attention_masks, start_positions, end_positions
    )
    train_size = int(train_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    logger.info(
        f"The input dataset has {len(dataset)} input samples, which have been split into {train_size} training "
        f"samples and {valid_size} validation samples."
    )
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))  # could do with shuffle=True instead?
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, sampler=SequentialSampler(valid_dataset))
    logger.info(f"There are {len(train_dataloader)} training batches and {len(valid_dataloader)} validation batches.")

    training_steps = training_epochs * len(train_dataloader)  # epochs * number of batches
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=training_steps)

    training_stats = {}
    for epoch in (range(training_epochs)):
        logger.info(f"Training epoch {epoch + 1} of {training_epochs}. Running training.")
        t_i = time()
        model.train()
        cumulative_train_loss_per_epoch = 0.
        for batch_num, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            logger.info(f"Running training batch {batch_num + 1} of {len(train_dataloader)}.")
            batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_start_positions, batch_end_positions = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
            model.zero_grad()
            #  model.zero_grad() and optimizer.zero_grad() are the same IF all model parameters are in that optimizer.
            #  It could be safer to call model.zero_grad() if you have two or more optimizers for one model.
            loss, start_logits, end_logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                token_type_ids=batch_token_type_ids,
                start_positions=batch_start_positions,
                end_positions=batch_end_positions
            )  # BertForQuestionAnswering uses CrossEntropyLoss by default, no need to calculate explicitly

            cumulative_train_loss_per_epoch += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            # clipping the norm of the gradients to 1.0 to help prevent the "exploding gradients" issues.
            optimizer.step()  # update model parameters
            lr_scheduler.step()  # update the learning rate

        average_training_loss_per_batch = cumulative_train_loss_per_epoch / len(train_dataloader)
        training_time = format_time(time() - t_i)
        logger.warning(f"Epoch {epoch + 1} took {training_time} to train.")
        logger.warning(f"Average training loss: {average_training_loss_per_batch}. \n Running validation.")

        t_i = time()
        model.eval()

        pred_start = torch.tensor([], dtype=torch.long, device=device)  # initialising tensors for storing results
        pred_end = torch.tensor([], dtype=torch.long, device=device)
        true_start = torch.tensor([], dtype=torch.long, device=device)
        true_end = torch.tensor([], dtype=torch.long, device=device)

        cumulative_eval_loss_per_epoch = 0
        cumulative_eval_accuracy_per_epoch = 0

        for batch_num, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            logger.info(f"Running validation batch {batch_num + 1} of {len(valid_dataloader)}.")
            batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_start_positions, batch_end_positions = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
            with torch.no_grad():
                loss, start_logits, end_logits = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_masks,
                    token_type_ids=batch_token_type_ids,
                    start_positions=batch_start_positions,
                    end_positions=batch_end_positions
                )
                cumulative_eval_loss_per_epoch += loss.item()
                # SHALL WE MOVE THE BELOW TO CPU AND NUMPY OR KEEP GPU AND PYTORCH?

                pred_start_positions = torch.argmax(start_logits, dim=1)
                pred_end_positions = torch.argmax(end_logits, dim=1)

                pred_start = torch.cat((pred_start, pred_start_positions))
                pred_end = torch.cat((pred_end, pred_end_positions))
                true_start = torch.cat((true_start, batch_start_positions))
                true_end = torch.cat((true_end, batch_end_positions))

        total_correct_start = int(sum(pred_start == true_start))
        total_correct_end = int(sum(pred_end == true_end))
        total_correct = total_correct_start + total_correct_end
        total_indices = len(true_start) + len(true_end)

        average_validation_accuracy_per_epoch = total_correct / total_indices
        average_validation_loss_per_batch = cumulative_eval_loss_per_epoch / len(valid_dataloader)
        valid_time = format_time(time() - t_i)
        logger.warning(f"Epoch {epoch + 1} took {valid_time} to validate.")
        logger.warning(f"Average validation loss: {average_validation_loss_per_batch}.")
        logger.warning(f"Average validation accuracy (out of 1): {average_validation_accuracy_per_epoch}.")

        training_stats[f"epoch_{epoch + 1}"] = {
            "training_loss": average_training_loss_per_batch,
            "valid_loss": average_validation_loss_per_batch,
            "valid_accuracy": average_validation_accuracy_per_epoch,
            "training_time": training_time,
            "valid_time": valid_time
        }
    if save_model_path is not None:
        torch.save(model, save_model_path)
    return training_stats
