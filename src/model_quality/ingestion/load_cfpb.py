from __future__ import annotations

import importlib.util
import shutil
import tempfile
import warnings
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from model_quality.simulation.scenarios import get_scenario

COLUMN_MAP = {
    'Product': 'product',
    'Sub-product': 'sub_product',
    'Issue': 'issue',
    'Sub-issue': 'sub_issue',
    'Consumer complaint narrative': 'consumer_complaint_narrative',
    'Company public response': 'company_public_response',
    'Company response to consumer': 'company_response_to_consumer',
    'Timely response?': 'timely_response',
    'Date received': 'date_received',
    'Date sent to company': 'date_sent_to_company',
    'State': 'state',
    'Submitted via': 'submitted_via',
    'Company': 'company',
}

NORMALIZED_REQUIRED = [
    'product', 'issue', 'consumer_complaint_narrative', 'company_response_to_consumer',
    'timely_response', 'date_received', 'state'
]

SUPPORTED_DATA_SUFFIXES = ('.csv', '.csv.gz', '.7z')


def _is_supported_data_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_DATA_SUFFIXES)


def discover_cfpb_path(data_dir: str | Path = 'data/raw') -> Path:
    """Find exactly one supported dataset in the local raw-data directory."""
    root = Path(data_dir)
    candidates = sorted(
        path
        for path in root.glob('*')
        if path.is_file() and not path.name.startswith('.') and _is_supported_data_file(path)
    )
    if not candidates:
        raise FileNotFoundError(
            f'No CFPB dataset found in {root}. Add one .csv, .csv.gz, or .7z file, '
            'pass --cfpb-path explicitly, or use --use-synthetic.'
        )
    if len(candidates) > 1:
        joined = ', '.join(path.name for path in candidates)
        raise ValueError(
            f'Multiple supported datasets found in {root}: {joined}. Pass --cfpb-path explicitly.'
        )
    return candidates[0]


def _validate_archive_members(names: list[str]) -> None:
    """Reject absolute or parent-traversal paths before archive extraction."""
    for raw_name in names:
        normalized = str(raw_name).replace('\\', '/')
        member = PurePosixPath(normalized)
        if member.is_absolute() or '..' in member.parts:
            raise ValueError(f'Unsafe archive member path: {raw_name!r}')


def _read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def _cached_archive_csv(path: Path, cache_dir: Path) -> Path:
    """Extract a 7z archive once and reuse the cached CSV on later runs."""
    stat = path.stat()
    cache_key = f'{path.stem}_{stat.st_size}_{stat.st_mtime_ns}'
    target = cache_dir / cache_key
    marker = target / '.complete'
    if marker.exists():
        csvs = sorted(target.rglob('*.csv'))
        if csvs:
            return csvs[0]
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    import py7zr
    with py7zr.SevenZipFile(path, mode='r') as archive:
        _validate_archive_members(archive.getnames())
        archive.extractall(target)
    csvs = sorted(target.rglob('*.csv'))
    if not csvs:
        shutil.rmtree(target, ignore_errors=True)
        raise FileNotFoundError('No CSV found inside .7z archive.')
    marker.write_text('complete', encoding='utf-8')
    return csvs[0]


def _duckdb_reservoir_sample_csv(path: Path, sample_size: int, random_state: int) -> pd.DataFrame:
    """Fast deterministic reservoir sampling for very large public CSV files.

    DuckDB performs the full-file reservoir scan in native code. The parser is
    configured defensively because public complaint exports can contain a small
    number of malformed records or quoted newlines.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            'DuckDB sampling requires duckdb. Install requirements.txt or use '
            '--sampling-strategy reservoir/head.'
        ) from exc

    con = duckdb.connect(database=':memory:')
    try:
        con.execute('PRAGMA threads=2')
        con.execute("PRAGMA memory_limit='2GB'")
        read_expr = (
            "read_csv_auto(?, header=true, all_varchar=true, quote='\"', escape='\"', "
            "strict_mode=false, null_padding=true, ignore_errors=true, "
            "parallel=false, sample_size=20000)"
        )
        query = f'''\
            SELECT *
            FROM {read_expr}
            WHERE length(coalesce("Consumer complaint narrative", '')) > 20
            USING SAMPLE reservoir({int(sample_size)} ROWS) REPEATABLE({int(random_state)})
        '''
        raw = con.execute(query, [str(path)]).fetchdf()
    finally:
        con.close()
    if raw.empty:
        raise ValueError('No usable complaint narratives found while DuckDB-sampling the public CSV.')
    result = normalize_cfpb(raw).head(sample_size).reset_index(drop=True)
    result.attrs['effective_sampling_strategy'] = 'duckdb'
    return result


def _resolve_sampling_strategy(path: Path, requested: str) -> str:
    if requested != 'auto':
        return requested
    if path.stat().st_size >= 100 * 1024 * 1024 and importlib.util.find_spec('duckdb') is not None:
        return 'duckdb'
    return 'reservoir'


def _reservoir_sample_csv(path: Path, sample_size: int, random_state: int, chunksize: int = 50000) -> pd.DataFrame:
    """Uniformly sample usable rows across a large CSV without loading it all.

    Each usable row receives an independent random priority; the globally
    smallest priorities are retained. Source-schema metadata is preserved so
    downstream quality checks can distinguish an absent source column from a
    normalized placeholder column.
    """
    rng = np.random.default_rng(random_state)
    reservoir = pd.DataFrame()
    source_columns: set[str] = set()
    for chunk in pd.read_csv(path, chunksize=max(1000, int(chunksize)), low_memory=False):
        normalized = normalize_cfpb(chunk)
        source_columns.update(normalized.attrs.get('source_normalized_columns', []))
        normalized['_sample_key'] = rng.random(len(normalized))
        reservoir = pd.concat([reservoir, normalized], ignore_index=True)
        if len(reservoir) > sample_size:
            reservoir = reservoir.nsmallest(sample_size, '_sample_key').reset_index(drop=True)
    if reservoir.empty:
        raise ValueError('No usable complaint narratives found while sampling the public CSV.')
    result = reservoir.nsmallest(min(sample_size, len(reservoir)), '_sample_key').drop(columns='_sample_key').reset_index(drop=True)
    result.attrs['source_normalized_columns'] = sorted(source_columns)
    result.attrs['source_required_columns_present'] = sorted(set(NORMALIZED_REQUIRED) & source_columns)
    result.attrs['effective_sampling_strategy'] = 'reservoir'
    return result


def load_cfpb(
    path: str | Path | None = None,
    sample_size: int = 5000,
    random_state: int = 7,
    use_synthetic: bool = False,
    scenario: str = 'nominal',
    scenario_config_path: str | Path = 'configs/monte_carlo.yaml',
    sampling_strategy: str = 'auto',
    source_chunksize: int = 50000,
    archive_cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load CFPB CSV or generate a scenario-aware synthetic CFPB-like sample.

    Supports .csv, .csv.gz, and .7z if py7zr is installed. The ``auto``
    strategy uses DuckDB for large files and pandas reservoir sampling for
    smaller files.
    """
    if use_synthetic:
        params = get_scenario(scenario, scenario_config_path)
        result = make_synthetic_cfpb(
            sample_size=sample_size,
            random_state=random_state,
            scenario_params=params,
        )
        result.attrs['effective_sampling_strategy'] = 'synthetic'
        return result
    if path is None:
        raise ValueError('A CFPB path is required unless --use-synthetic is set.')

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'CFPB path not found: {path}')

    if path.suffix.lower() == '.7z':
        try:
            import py7zr  # noqa: F401
        except ImportError as e:
            raise ImportError('Install py7zr to read .7z files, or extract the CSV first.') from e

        def sample_csv(csv_path: Path) -> pd.DataFrame:
            effective = _resolve_sampling_strategy(csv_path, sampling_strategy)
            if effective == 'head':
                result = normalize_cfpb(
                    _read_csv(csv_path, nrows=max(sample_size * 2, sample_size))
                ).head(sample_size)
                result.attrs['effective_sampling_strategy'] = 'head'
                return result
            if effective == 'duckdb':
                return _duckdb_reservoir_sample_csv(csv_path, sample_size, random_state)
            if effective == 'reservoir':
                return _reservoir_sample_csv(csv_path, sample_size, random_state, source_chunksize)
            raise ValueError(
                f'Unknown sampling_strategy={sampling_strategy!r}; '
                'use auto, duckdb, reservoir, or head.'
            )

        if archive_cache_dir is not None:
            csv_path = _cached_archive_csv(path, Path(archive_cache_dir))
            df = sample_csv(csv_path)
        else:
            with tempfile.TemporaryDirectory() as td:
                with py7zr.SevenZipFile(path, mode='r') as z:
                    _validate_archive_members(z.getnames())
                    z.extractall(td)
                csvs = sorted(Path(td).rglob('*.csv'))
                if not csvs:
                    raise FileNotFoundError('No CSV found inside .7z archive.')
                df = sample_csv(csvs[0])
    else:
        effective = _resolve_sampling_strategy(path, sampling_strategy)
        if effective == 'head':
            df = normalize_cfpb(_read_csv(path, nrows=max(sample_size * 2, sample_size))).head(sample_size)
            df.attrs['effective_sampling_strategy'] = 'head'
        elif effective == 'duckdb':
            df = _duckdb_reservoir_sample_csv(path, sample_size, random_state)
        elif effective == 'reservoir':
            df = _reservoir_sample_csv(path, sample_size, random_state, source_chunksize)
        else:
            raise ValueError(
                f'Unknown sampling_strategy={sampling_strategy!r}; '
                'use auto, duckdb, reservoir, or head.'
            )

    attrs = dict(df.attrs)
    result = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    result.attrs.update(attrs)
    if sampling_strategy == 'auto' and result.attrs.get('effective_sampling_strategy') == 'reservoir':
        warnings.warn(
            'DuckDB was unavailable or the file was small; using pandas reservoir sampling.',
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def normalize_cfpb(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns}).copy()
    df.columns = [str(c).strip().replace(' ', '_').replace('-', '_').replace('?', '').lower() for c in df.columns]
    source_normalized_columns = sorted(map(str, df.columns))
    source_required_present = sorted(set(NORMALIZED_REQUIRED) & set(source_normalized_columns))
    for c in NORMALIZED_REQUIRED:
        if c not in df.columns:
            df[c] = np.nan
    df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].fillna('').astype(str)
    df['product'] = df['product'].fillna('Unknown').astype(str)
    df['issue'] = df['issue'].fillna('Unknown').astype(str)
    df['company_response_to_consumer'] = df['company_response_to_consumer'].fillna('Unknown').astype(str)
    df['timely_response'] = df['timely_response'].fillna('Unknown').astype(str)
    df['state'] = df['state'].fillna('NA').astype(str)
    df['date_received'] = pd.to_datetime(df['date_received'], errors='coerce', format='mixed')
    df = df[df['consumer_complaint_narrative'].str.len() > 20].copy()
    if df.empty:
        raise ValueError('No usable complaint narratives found after normalization.')
    result = df.reset_index(drop=True)
    result.attrs['source_normalized_columns'] = source_normalized_columns
    result.attrs['source_required_columns_present'] = source_required_present
    return result


def _shifted_product_probs(base_probs: np.ndarray, strength: float) -> np.ndarray:
    """Shift mass toward later-index products while preserving a valid distribution."""
    strength = float(np.clip(strength, 0.0, 0.8))
    tilt = np.array([-0.18, -0.10, -0.04, 0.12, 0.20], dtype=float)
    probs = np.clip(base_probs + strength * tilt, 0.02, None)
    return probs / probs.sum()


def make_synthetic_cfpb(
    sample_size: int = 2500,
    random_state: int = 7,
    scenario_params: dict | None = None,
) -> pd.DataFrame:
    """Generate noisy CFPB-like data for smoke tests and Monte Carlo validation.

    The scenario controls ambiguity, distractors, label conflicts, missingness,
    temporal product shift, and stale-evidence language. The same seed and
    scenario produce the same data; independent child seeds are used by the
    Monte Carlo runner.
    """
    p = scenario_params or {}
    ambiguity_rate = float(p.get('ambiguity_rate', 0.12))
    distractor_rate = float(p.get('distractor_rate', 0.12))
    stale_rate = float(p.get('stale_evidence_rate', 0.08))
    conflict_rate = float(p.get('narrative_conflict_rate', 0.03))
    label_noise_rate = float(p.get('label_noise_rate', 0.02))
    missing_state_rate = float(p.get('missing_state_rate', 0.005))
    temporal_shift = float(p.get('temporal_product_shift', 0.03))

    rng = np.random.default_rng(random_state)
    products = ['Credit card', 'Checking or savings account', 'Credit reporting', 'Mortgage', 'Money transfer']
    issues = ['Incorrect information', 'Problem with purchase', 'Closing an account', 'Trouble during payment process', 'Managing account']
    responses = ['Closed with explanation', 'Closed with monetary relief', 'Closed with non-monetary relief', 'In progress']
    states = ['NJ', 'NY', 'CA', 'TX', 'FL', 'IL']
    base_product_probs = np.array([0.22, 0.18, 0.28, 0.17, 0.15], dtype=float)
    shifted_probs = _shifted_product_probs(base_product_probs, temporal_shift)
    product_terms = {
        'Credit card': ['card charge', 'billing statement', 'merchant dispute', 'interest fee', 'credit line'],
        'Checking or savings account': ['deposit account', 'ATM withdrawal', 'debit transaction', 'checking balance', 'bank account'],
        'Credit reporting': ['credit report', 'score dropped', 'bureau record', 'tradeline', 'dispute investigation'],
        'Mortgage': ['escrow account', 'loan servicer', 'monthly payment', 'foreclosure notice', 'mortgage statement'],
        'Money transfer': ['wire transfer', 'remittance', 'recipient did not receive funds', 'transfer receipt', 'international payment'],
    }
    issue_terms = {
        'Incorrect information': ['incorrect information', 'wrong record', 'mismatch in the file', 'inaccurate balance'],
        'Problem with purchase': ['purchase problem', 'merchant issue', 'refund not posted', 'goods were not received'],
        'Closing an account': ['closing request', 'account closure', 'unable to close', 'closure confirmation'],
        'Trouble during payment process': ['payment failed', 'processing delay', 'autopay problem', 'late payment notice'],
        'Managing account': ['account access', 'customer service', 'document upload', 'statement review'],
    }
    generic_noise = [
        'I contacted customer service multiple times and received different explanations.',
        'The company said the matter was resolved but the documents do not support that conclusion.',
        'I uploaded screenshots, notices, and correspondence but the response did not address them.',
        'The issue has affected my ability to understand the account status and next steps.',
    ]
    rows = []
    base = pd.Timestamp('2022-05-01')
    horizon_days = 1460
    for _ in range(sample_size):
        day = int(rng.integers(0, horizon_days))
        use_shifted = day >= int(0.70 * horizon_days)
        probs = shifted_probs if use_shifted else base_product_probs
        latent_product = str(rng.choice(products, p=probs))
        issue = str(rng.choice(issues))
        term = str(rng.choice(product_terms[latent_product]))
        issue_term = str(rng.choice(issue_terms[issue]))

        distractor_product = str(rng.choice([x for x in products if x != latent_product]))
        distractor = str(rng.choice(product_terms[distractor_product])) if rng.random() < distractor_rate else ''
        if rng.random() < (1.0 - ambiguity_rate):
            opening = f'I am filing a complaint about a {term} because of {issue_term}.'
        else:
            opening = f'I need help with {issue_term}; the records and explanations do not match.'
        if distractor:
            opening += f' A separate note also referenced {distractor}, which made the response confusing.'
        text = ' '.join([opening, str(rng.choice(generic_noise))])
        if rng.random() < stale_rate:
            text += ' The evidence appears stale or incomplete, and the final response may be unsupported.'
        if rng.random() < conflict_rate:
            text += f' The records conflict and alternately label the case as {distractor_product}.'

        observed_product = latent_product
        if rng.random() < label_noise_rate:
            observed_product = str(rng.choice([x for x in products if x != latent_product]))

        timely_prob = 0.90
        if 'stale or incomplete' in text or issue == 'Trouble during payment process':
            timely_prob -= 0.12
        if rng.random() < conflict_rate:
            timely_prob -= 0.08
        timely_prob = float(np.clip(timely_prob, 0.55, 0.97))
        rows.append({
            'product': observed_product,
            'latent_product': latent_product,
            'issue': issue,
            'consumer_complaint_narrative': text,
            'company_response_to_consumer': rng.choice(responses, p=[0.63, 0.09, 0.13, 0.15]),
            'timely_response': rng.choice(['Yes', 'No'], p=[timely_prob, 1 - timely_prob]),
            'date_received': base + pd.Timedelta(days=day),
            'state': rng.choice(states),
            'company': f'Company_{rng.integers(1, 60)}',
            'submitted_via': rng.choice(['Web', 'Phone', 'Referral', 'Email'], p=[0.72, 0.12, 0.10, 0.06]),
        })
    df = pd.DataFrame(rows)
    if sample_size > 0 and missing_state_rate > 0:
        n_missing = min(sample_size, max(1, int(round(missing_state_rate * sample_size))))
        missing_idx = rng.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_idx, 'state'] = 'NA'
    df.attrs['source_normalized_columns'] = sorted(map(str, df.columns))
    df.attrs['source_required_columns_present'] = sorted(set(NORMALIZED_REQUIRED) & set(df.columns))
    return df
