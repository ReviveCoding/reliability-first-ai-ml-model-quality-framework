from model_quality.retrieval.cross_encoder_reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self):
        self.calls = 0
        self.last_pairs = None

    def predict(self, pairs, batch_size, show_progress_bar):
        self.calls += 1
        self.last_pairs = list(pairs)
        return [float(len(doc)) for _, doc in pairs]


def test_rerank_many_uses_one_batched_predict_call():
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = 'fake'
    reranker.device = 'cpu'
    reranker.batch_size = 4
    reranker.model = FakeCrossEncoder()
    reranker.note = ''

    results = reranker.rerank_many(
        ['q1', 'q2'],
        ['a', 'long document', 'medium'],
        [[0, 1], [0, 2]],
    )
    assert reranker.model.calls == 1
    assert len(reranker.model.last_pairs) == 4
    assert results[0].rankings[0][0] == 1
    assert results[1].rankings[0][0] == 2
