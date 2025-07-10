import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to training json file')
    parser.add_argument('--output', required=True, help='output feature npy file')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='sentence transformer model')
    args = parser.parse_args()

    with open(args.data, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    sentences = [item[3] for item in data]

    st_model = SentenceTransformer(args.model)
    feats = st_model.encode(sentences, batch_size=64, show_progress_bar=True)
    np.save(args.output, feats)
    print('Saved features to', args.output)


if __name__ == '__main__':
    main()
