"""write transcripts into a file: a transcript sentence per line"""
import json
import os
from pathlib import Path
import logging
from tqdm.auto import tqdm
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from argparse import ArgumentParser
from multiprocessing import Pool

ROOT_FOLDER = Path().resolve()

# @tokenizer: the tokenizer to tokenize words in a single sentence
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)



def raw_data_processer(lines):
    """Process raw lines
    effects: append single word onto the last sentence in its list
    input:
    @lines: input lines
     [ "hi everyone welcome back to my channel",
     "I hope you guys are having an awesome", "day",
     "the healthy grocery girl is also my friend Meghan Roosevelt"]

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
    parser.add_argument('--raw_file', type=Path, default=default_transcript_path)
    parser.add_argument('--output_dir', type=Path, default=default_output_path)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--chunksize', type=int, default=128)
    args = parser.parse_args()

    logging.info('Reading the whole transcript file')
    with open(args.raw_file, 'r') as istream:
        transcripts = json.load(istream)

    if not os.path.exists(args.output_dir):
        logging.info('Created dir %s' % args.output_dir)
        os.makedirs(args.output_dir)
    output_file = os.path.join(args.output_dir, 'transcripts.txt')
    ostream = open(output_file, 'w')
    results = list()
    lines = list(transcripts.values())
    logging.info('Start nultiprocess: workers: {}, chunksize: {}'
                 .format(args.num_workers, args.chunksize))

    pool = Pool(processes=args.num_workers)
    for sentences in tqdm(pool.imap(raw_data_processer, lines, args.chunksize), desc='Preprocess data'):
        results.append(sentences)

    for sentences in tqdm(results, desc='Writing to file'):
        if len(sentences) <= 1:
            logging.info('Skipped miniclip with <=1 sentence: {}'.format(sentences))
            continue
        for sentence in sentences:
            ostream.write('%s\n' % sentence)
        ostream.write('\n')
        ostream.flush()  # flush to file
    ostream.close()

    logging.info('Save processed file into {}'.format(output_file))


if __name__ == '__main__':
    main()