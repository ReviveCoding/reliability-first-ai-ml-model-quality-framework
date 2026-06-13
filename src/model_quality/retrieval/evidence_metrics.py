from __future__ import annotations

import pandas as pd

from .bm25_retriever import SimpleBM25
from .cross_encoder_reranker import CrossEncoderReranker


def _build_document_text(df: pd.DataFrame) -> list[str]:
    parts = []
    for _, r in df.fillna('').iterrows():
        # Retrieval relies on complaint evidence rather than exact gold labels.
        parts.append(' '.join([
            f"company response: {r.get('company_response_to_consumer','')}",
            f"narrative: {r.get('consumer_complaint_narrative','')}",
        ]))
    return parts


PRODUCT_HINTS = {
    'Credit card': 'card billing merchant charge interest fee credit line',
    'Checking or savings account': 'bank account debit ATM deposit checking balance',
    'Credit reporting': 'credit report bureau score tradeline dispute investigation',
    'Mortgage': 'mortgage escrow loan servicer foreclosure monthly payment',
    'Money transfer': 'wire transfer remittance recipient funds international payment',
}
ISSUE_HINTS = {
    'Incorrect information': 'incorrect wrong inaccurate mismatch record balance',
    'Problem with purchase': 'purchase merchant refund goods not received',
    'Closing an account': 'close closure request confirmation account',
    'Trouble during payment process': 'payment failed processing delay autopay late notice',
    'Managing account': 'account access customer service document statement review',
}


def _task_query(product: str, issue: str) -> str:
    product_hint = PRODUCT_HINTS.get(str(product), str(product))
    issue_hint = ISSUE_HINTS.get(str(issue), str(issue))
    return (
        f"Find complaint evidence about {product_hint}. "
        f"The issue involves {issue_hint}. Prioritize detailed supporting narratives."
    )


def build_evidence_cases(df: pd.DataFrame, n_cases: int = 100) -> pd.DataFrame:
    sample = df.head(n_cases).copy().reset_index(drop=False).rename(columns={'index': 'source_doc_id'})
    sample['query'] = [_task_query(p, i) for p, i in zip(sample['product'].astype(str), sample['issue'].astype(str))]
    sample['gold_product'] = sample['latent_product'].astype(str) if 'latent_product' in sample.columns else sample['product'].astype(str)
    return sample[['query','source_doc_id','consumer_complaint_narrative','product','gold_product','issue']]


def _is_relevant(doc_row: pd.Series, gold_product: str, issue: str) -> bool:
    candidate_product = doc_row.get('latent_product', doc_row.get('product', ''))
    return str(candidate_product) == str(gold_product) and str(doc_row.get('issue', '')) == str(issue)


def _ranking_metrics(df0: pd.DataFrame, ids: list[int], gold_product: str, issue: str) -> tuple[bool, int | None, float]:
    flags = [_is_relevant(df0.iloc[i], gold_product, issue) for i in ids]
    hit = any(flags)
    rank = flags.index(True) + 1 if hit else None
    precision = sum(flags) / max(1, len(flags))
    return hit, rank, precision


def _aggregate(detail: pd.DataFrame, prefix: str) -> dict:
    hit_col = f'{prefix}_hit_at_k'
    rank_col = f'{prefix}_rank'
    precision_col = f'{prefix}_precision_at_k'
    recall = float(detail[hit_col].mean()) if len(detail) else 0.0
    mrr = float(detail[rank_col].dropna().map(lambda r: 1.0 / r).mean()) if detail[rank_col].notna().any() else 0.0
    precision = float(detail[precision_col].mean()) if len(detail) else 0.0
    return {'recall_at_5': recall, 'mrr': mrr, 'context_precision': precision}


def evaluate_evidence_quality(
    df: pd.DataFrame,
    use_cross_encoder: bool = False,
    top_k_bm25: int = 20,
    top_k_final: int = 5,
    device: str = 'auto',
    cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    cross_encoder_batch_size: int = 32,
) -> tuple[dict, pd.DataFrame]:
    df0 = df.reset_index(drop=True)
    docs = _build_document_text(df0)
    bm25 = SimpleBM25(docs)
    cases = build_evidence_cases(df0, min(150, len(df0)))

    queries = cases['query'].astype(str).tolist()
    bm25_candidates: list[list[int]] = []
    for query in queries:
        retrieved = bm25.retrieve(query, top_k=min(top_k_bm25, len(docs)))
        bm25_candidates.append([i for i, _ in retrieved])

    reranker = CrossEncoderReranker(
        model_name=cross_encoder_model,
        device=device,
        batch_size=cross_encoder_batch_size,
    ) if use_cross_encoder else None
    if reranker is not None:
        rerank_results = reranker.rerank_many(queries, docs, bm25_candidates)
    else:
        rerank_results = [None] * len(cases)

    rows = []
    notes: list[str] = []
    ce_used = False
    for case_idx, (_, row) in enumerate(cases.iterrows()):
        baseline_ids = bm25_candidates[case_idx][:top_k_final]
        rr = rerank_results[case_idx]
        if rr is not None:
            ce_used = ce_used or rr.used_cross_encoder
            if rr.note:
                notes.append(rr.note)
            final_ids = [i for i, _ in rr.rankings[:top_k_final]]
        else:
            final_ids = baseline_ids

        bm25_hit, bm25_rank, bm25_precision = _ranking_metrics(
            df0, baseline_ids, row['gold_product'], row['issue']
        )
        final_hit, final_rank, final_precision = _ranking_metrics(
            df0, final_ids, row['gold_product'], row['issue']
        )
        rows.append({
            'query': row['query'],
            'source_doc_id': int(row['source_doc_id']),
            'product': row['product'],
            'gold_product': row['gold_product'],
            'issue': row['issue'],
            'bm25_hit_at_k': bm25_hit,
            'bm25_rank': bm25_rank,
            'bm25_precision_at_k': bm25_precision,
            'bm25_retrieved_ids': '|'.join(map(str, baseline_ids)),
            'final_hit_at_k': final_hit,
            'final_rank': final_rank,
            'final_precision_at_k': final_precision,
            'final_retrieved_ids': '|'.join(map(str, final_ids)),
            # Backward-compatible aliases used by the review-queue and v5 tests.
            'hit_at_k': final_hit,
            'rank': final_rank,
            'precision_at_k': final_precision,
            'retrieved_ids': '|'.join(map(str, final_ids)),
        })

    detail = pd.DataFrame(rows)
    baseline = _aggregate(detail, 'bm25') if len(detail) else {'recall_at_5': 0.0, 'mrr': 0.0, 'context_precision': 0.0}
    final = _aggregate(detail, 'final') if len(detail) else baseline.copy()
    metrics = {
        'recall_at_5': final['recall_at_5'],
        'mrr': final['mrr'],
        'context_precision': final['context_precision'],
        'unsupported_claim_rate': float(1.0 - final['recall_at_5']),
        'bm25_recall_at_5': baseline['recall_at_5'],
        'bm25_mrr': baseline['mrr'],
        'bm25_context_precision': baseline['context_precision'],
        'rerank_recall_uplift': float(final['recall_at_5'] - baseline['recall_at_5']),
        'rerank_mrr_uplift': float(final['mrr'] - baseline['mrr']),
        'rerank_context_precision_uplift': float(final['context_precision'] - baseline['context_precision']),
        'cross_encoder_requested': bool(use_cross_encoder),
        'used_cross_encoder': bool(ce_used),
        'note': notes[0] if notes else '',
        'case_count': int(len(detail)),
    }
    return metrics, detail
