# NSP-fine-tune
Fine-tune BERT using Next Sentence Prediction Loss

~~~
Usage:
./bin/data preprocess # preprocess data
./bin/data pregenerate # add random mask and datasets for all epochs
Directory Tree:
.
├── README.md
├── bin
│   ├── data
│   ├── log
│   ├── pregenerate
│   ├── preprocess
│   └── train
├── checkpoints
├── data
│   ├── interim
│   │   └── transcripts.txt
│   ├── processed
│   └── raw
│       └── transcripts_sentences.json
├── next_sentence_pred_finetune
│   ├── dataset.py
│   ├── evaluate.py
│   ├── finetune_bert.py
│   ├── inference.py
│   ├── pregenerate_data.py
│   ├── preprocess_data.py
│   ├── simple_lm_finetuning.py
│   ├── train.py
│   └── utils.py
├── sbatch
│   └── train
├── setup.py
└── tests

9 directories, 19 files
~~~
