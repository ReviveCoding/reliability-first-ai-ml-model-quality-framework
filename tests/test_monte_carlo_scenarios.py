from model_quality.ingestion.load_cfpb import make_synthetic_cfpb
from model_quality.simulation.scenarios import get_scenario, spawn_run_seeds


def test_spawn_run_seeds_reproducible_and_distinct():
    a = spawn_run_seeds(123, 5)
    b = spawn_run_seeds(123, 5)
    assert a == b
    assert len(set(a)) == 5


def test_scenario_severity_changes_synthetic_difficulty():
    nominal = get_scenario('nominal')
    severe = get_scenario('severe')
    a = make_synthetic_cfpb(500, 42, nominal)
    b = make_synthetic_cfpb(500, 42, severe)
    # Severe scenarios should create more observed-vs-latent label conflicts.
    nominal_noise = (a['product'] != a['latent_product']).mean()
    severe_noise = (b['product'] != b['latent_product']).mean()
    assert severe_noise > nominal_noise
    assert (b['state'] == 'NA').mean() > (a['state'] == 'NA').mean()
