"""Script to build a semantic FAISS index from a strategy corpus.

Usage:
    python scripts/build_semantic_index.py --input data/strategies.csv --text-col strategy --out models/semantic_index --method flat

Supports 'flat' (IndexFlatIP) and 'ivfpq' (IndexIVFPQ) methods.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
import numpy as np

from src.semantic_recommender import SemanticRecommender


def read_corpus(path: str, text_col: str = 'strategy') -> List[str]:
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
        if text_col in df.columns:
            return df[text_col].astype(str).tolist()
        else:
            # fallback to first text column
            return df.iloc[:, 0].astype(str).tolist()
    else:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Path to corpus (csv or txt)')
    p.add_argument('--text-col', default='strategy', help='CSV column containing strategy text')
    p.add_argument('--out', '-o', default='models/semantic_index', help='Output index prefix')
    p.add_argument('--method', choices=['flat', 'ivfpq'], default='flat', help='Indexing method')
    p.add_argument('--nlist', type=int, default=100, help='Number of IVF clusters (for ivfpq)')
    p.add_argument('--m', type=int, default=8, help='Number of PQ subquantizers (for ivfpq)')
    args = p.parse_args()

    corpus = read_corpus(args.input, text_col=args.text_col)
    sr = SemanticRecommender()

    if args.method == 'flat':
        sr.index_from_corpus(corpus, ids=[f"s{i}" for i in range(len(corpus))])
    else:
        # For large corpora use IVF+PQ - SemanticRecommender does not have ivfpq path yet
        # We'll implement a simple alternative here using the SentenceTransformer directly
        embs = sr.embed_texts(corpus)
        embs = (embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)).astype('float32')
        import faiss
        d = embs.shape[1]
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, args.nlist, args.m, 8)
        # train
        index.train(embs)
        index.add(embs)
        sr.index = index
        sr.ids = [f"s{i}" for i in range(len(corpus))]

    sr.save_index(args.out)
    print(f"Saved index to {args.out}.index and metadata to {args.out}.meta.joblib")


if __name__ == '__main__':
    main()
