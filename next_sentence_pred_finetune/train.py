"""Finetune using Next Sentence Prediction loss"""
from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from pathlib import Path
import torch
import logging
import json
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.modeling_bert import BertForPreTraining
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from torch.utils.tensorboard import SummaryWriter

from next_sentence_pred_finetune.evaluate import evaluate
from next_sentence_pred_finetune.dataset import PregeneratedDataset

logger = logging.getLogger(__name__)


def train(args, model, optimizer, scheduler):
    tb_writer = SummaryWriter(args.label)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {args.total_train_examples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", args.num_train_optimization_steps)
    # model.train()
    model.zero_grad() #
    for epoch in range(args.epochs):
        model.train()
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=args.num_data_epochs, reduce_memory=args.reduce_memory)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
                outputs = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)
                loss = outputs[0]
                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                pbar.update(1)
                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                tb_writer.add_scalar('mean_loss', mean_loss, global_step)
                pbar.set_postfix_str(f"mean_loss: {mean_loss:.5f}")
                logging.info(f"training_loss: {mean_loss:.5f}")
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        eval_loss = evaluate(args, epoch, model, tokenizer)
        tb_writer.add_scalar('eval_loss', eval_loss, epoch)
        model.save_pretrained(args.output_dir / f"epoch_{epoch}")
        tokenizer.save_pretrained(args.output_dir / f"epoch_{epoch}")


