import pandas as pd

from model_quality.features.product_taxonomy import canonicalize_product_taxonomy


def test_canonical_taxonomy_preserves_raw_and_merges_historical_labels(tmp_path):
    config = tmp_path / 'taxonomy.yaml'
    config.write_text(
        """taxonomy_version: 1
target_column: product
raw_column: product_raw
mapping:
  Old credit report: Credit reporting
  Credit card: Card family
  Prepaid card: Card family
""",
        encoding='utf-8',
    )
    frame = pd.DataFrame({'product': ['Old credit report', 'Credit card', 'Prepaid card', 'Mortgage']})
    out, meta = canonicalize_product_taxonomy(frame, config)

    assert out['product_raw'].tolist() == frame['product'].tolist()
    assert out['product'].tolist() == ['Credit reporting', 'Card family', 'Card family', 'Mortgage']
    assert meta['changed_row_count'] == 3
    assert meta['raw_class_count'] == 4
    assert meta['canonical_class_count'] == 3
