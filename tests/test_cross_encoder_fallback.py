from model_quality.retrieval.cross_encoder_reranker import CrossEncoderReranker


def test_reranker_handles_invalid_and_empty_candidates_without_crashing():
    reranker = CrossEncoderReranker(device='unavailable-device-name')
    empty = reranker.rerank('query', ['doc'], [])
    assert empty.rankings == []
    result = reranker.rerank('query', ['doc a', 'doc b'], [-1, 1, 99])
    assert [idx for idx, _ in result.rankings] == [1]
