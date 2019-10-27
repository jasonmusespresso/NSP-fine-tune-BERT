from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from multiprocessing import Pool
from pytorch_transformers.tokenization_bert import BertTokenizer
from random import random

from next_sentence_pred_finetune.dataset import DocumentDatabase
from next_sentence_pred_finetune.utils import create_training_file, create_evaluating_file


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dev_frac", type=float, default=0.2,
                        help='The probability that a document is put in to the evaluation set.')
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual-uncased", "bert-base-chinese", "bert-base-multilingual-cased"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")

    args = parser.parse_args()

    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError("Cannot use multiple workers while reducing memory")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())
    with DocumentDatabase(reduce_memory=args.reduce_memory) as train_docs, \
            DocumentDatabase(reduce_memory=args.reduce_memory) as dev_docs:
        with args.train_corpus.open() as f:
            doc = []
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                line = line.strip()
                if line == "":
                    train_docs.add_document(doc) if random() < args.dev_frac else dev_docs.add_document(doc)
                    doc = []
                else:
                    tokens = tokenizer.tokenize(line)
                    doc.append(tokens)
            if doc:
                train_docs.add_document(doc) if random() < args.dev_frac else dev_docs.add_document(doc)
                # If the last doc didn't end on a newline, make sure it still gets added
        if len(train_docs) <= 1 or len(dev_docs) <= 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")

        args.output_dir.mkdir(exist_ok=True)

        if args.num_workers > 1:
            writer_workers = Pool(min(args.num_workers, args.epochs_to_generate))
            # training docs
            arguments = [(train_docs, vocab_list, args, idx) for idx in range(args.epochs_to_generate)]
            writer_workers.starmap(create_training_file, arguments)
            # evaluating docs
            arguments = [(dev_docs, vocab_list, args, idx) for idx in range(args.epochs_to_generate)]
            writer_workers.starmap(create_evaluating_file, arguments)
        else:
            for epoch in trange(args.epochs_to_generate, desc="Epoch"):
                create_training_file(train_docs, vocab_list, args, epoch)
                create_evaluating_file(dev_docs, vocab_list, args, epoch)


if __name__ == '__main__':
    main()
