from model_quality.retrieval.bm25_retriever import SimpleBM25


def test_bm25_retrieves_expected_doc():
    docs = ['credit card incorrect information', 'mortgage closing problem', 'money transfer delay']
    bm25 = SimpleBM25(docs)
    res = bm25.retrieve('credit card information', top_k=1)
    assert res[0][0] == 0
