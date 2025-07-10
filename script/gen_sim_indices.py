import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='training json file')
    parser.add_argument('--output', required=True, help='output npy with top-k indices')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    parser.add_argument('--feat-output', help='optional path to save sentence features')
    args = parser.parse_args()

    with open(args.data, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    sentences = [item[3] for item in data]

    st_model = SentenceTransformer(args.model)
    feats = st_model.encode(sentences, batch_size=64, show_progress_bar=True)

    norm = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats = feats / norm

    sims = feats @ feats.T
    order = np.argsort(-sims, axis=1)

    n = sims.shape[0]
    indices = []
    for i in range(n):
        idxs = [j for j in order[i] if j != i][:args.topk]
        indices.append(idxs)
    indices = np.array(indices, dtype=np.int32)
    np.save(args.output, indices)
    print('Saved indices to', args.output)
    if args.feat_output:
        np.save(args.feat_output, feats)
        print('Saved features to', args.feat_output)


if __name__ == '__main__':
    main()
