from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RerankResult:
    used_cross_encoder: bool
    device: str
    rankings: list[tuple[int, float]]
    note: str = ''


def detect_device(device='auto') -> str:
    requested = str(device).lower()
    try:
        import torch
    except Exception:
        return 'cpu' if requested == 'auto' else requested
    if requested == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    if requested.startswith('cuda'):
        return 'cuda' if torch.cuda.is_available() else 'unavailable-cuda'
    if requested == 'mps':
        available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        return 'mps' if available else 'unavailable-mps'
    return 'cpu'


class CrossEncoderReranker:
    """Lazy-load one optional second-stage reranker per evaluation run."""

    def __init__(
        self,
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        device: str = 'auto',
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = detect_device(device)
        self.batch_size = max(1, int(batch_size))
        self.model = None
        self.note = ''
        if self.device.startswith('unavailable-'):
            self.note = f'Requested device {device!r} is unavailable; BM25 fallback ranking used.'
            return
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, device=self.device)
        except Exception as exc:
            self.note = f'CrossEncoder unavailable; BM25 fallback ranking used: {type(exc).__name__}: {exc}'

    @property
    def available(self) -> bool:
        return self.model is not None

    def rerank_many(
        self,
        queries: list[str],
        docs: list[str],
        candidate_lists: list[list[int]],
    ) -> list[RerankResult]:
        """Rerank all query-candidate pairs in one batched model call.

        Batching avoids hundreds of small GPU launches during evidence
        evaluation while preserving one independently sorted result per query.
        """
        if len(queries) != len(candidate_lists):
            raise ValueError('queries and candidate_lists must have the same length.')

        sanitized: list[list[int]] = [
            [int(idx) for idx in candidates if 0 <= int(idx) < len(docs)]
            for candidates in candidate_lists
        ]
        if self.model is None:
            return [
                RerankResult(
                    False,
                    self.device,
                    [(idx, float(len(candidates) - rank)) for rank, idx in enumerate(candidates)],
                    note=self.note if candidates else 'No valid candidate documents were provided.',
                )
                for candidates in sanitized
            ]

        flat_pairs: list[tuple[str, str]] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for query, candidates in zip(queries, sanitized):
            start = cursor
            flat_pairs.extend((query, docs[idx]) for idx in candidates)
            cursor += len(candidates)
            offsets.append((start, cursor))

        if flat_pairs:
            raw_scores = self.model.predict(
                flat_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            scores = [float(score) for score in raw_scores]
        else:
            scores = []

        results: list[RerankResult] = []
        for candidates, (start, end) in zip(sanitized, offsets):
            rankings = sorted(
                zip(candidates, scores[start:end]),
                key=lambda item: item[1],
                reverse=True,
            )
            results.append(RerankResult(True, self.device, rankings))
        return results

    def rerank(self, query: str, docs: list[str], candidate_indices: list[int]) -> RerankResult:
        return self.rerank_many([query], docs, [candidate_indices])[0]


def rerank(
    query: str,
    docs: list[str],
    candidate_indices: list[int],
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    device: str = 'auto',
) -> RerankResult:
    return CrossEncoderReranker(model_name=model_name, device=device).rerank(query, docs, candidate_indices)
