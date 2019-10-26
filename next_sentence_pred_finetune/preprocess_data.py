"""write transcripts into a file: a transcript sentence per line"""
import json
import os
from pathlib import Path
from tqdm.auto import tqdm
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from argparse import ArgumentParser

ROOT_FOLDER = Path().resolve()


def raw_data_processer(lines, tokenizer=None):
    """Process raw lines
    effects: append single word onto the last sentence in its list
    input:
    @lines: input lines
     [ "hi everyone welcome back to my channel",
     "I hope you guys are having an awesome", "day",
     "the healthy grocery girl is also my friend Meghan Roosevelt"]
    @tokenizer: the tokenizer to tokenize words in a single sentence

    output:
    [ "hi everyone welcome back to my channel",
     "I hope you guys are having an awesome day",
     "the healthy grocery girl is also my friend Meghan Roosevelt"]
    """
    results = list()
    for line in lines:
        # tokenizer with spacy tokenizer
        line = line.strip()
        if len(tokenizer(line)) > 1:
            results.append(line)
        else:
            results[-1] = results[-1] + ' ' + line
    return results




def main():
    parser = ArgumentParser()
    default_transcript_path = os.path.join(ROOT_FOLDER, 'data', 'raw', 'transcripts_sentences.json')
    default_output_path = os.path.join(ROOT_FOLDER, 'data', 'processed')
    parser.add_argument('--raw_data', type=Path, default=default_transcript_path)
    parser.add_argument('--output_dir', type=Path, default=default_output_path)
    args = parser.parse_args()

    with open(args.transcript_path, 'r') as istream:
        transcripts = json.load(istream)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for _, lines in tqdm(transcripts.items(), desc='Preprocess Transcripts'):
        nlp = English()
        tokenizer = nlp.Defaults.create_tokenizer(nlp)
        lines = raw_data_processer(lines)


if __name__ == '__main__':
    main()