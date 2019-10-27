"""Finetune using Next Sentence Prediction loss"""
from __future__ import absolute_import, division, print_function

import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


from next_sentence_pred_finetune.dataset import PregeneratedDataset

logger = logging.getLogger(__name__)


def evaluate(args, epoch, model, tokenizer):
    model.eval()
    epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                        num_data_epochs=args.num_data_epochs, reduce_memory=args.reduce_memory, training=False)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(epoch_dataset)
    else:
        eval_sampler = DistributedSampler(epoch_dataset)

    eval_dataloader = DataLoader(epoch_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)
    eval_loss = 0
    nb_eval_steps = 0
    with tqdm(total=len(eval_dataloader), desc=f"Epoch {epoch}") as pbar:
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
            outputs = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)
            loss = outputs[0]
            # loss, prediction_scores, seq_relationship_score, hidden_states, attentions

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            eval_loss += loss.item()
            nb_eval_steps += 1
            pbar.update(1)
    eval_loss = eval_loss / nb_eval_steps
    logging.info(f"eval_loss: {eval_loss:.5f}")
    return eval_loss
