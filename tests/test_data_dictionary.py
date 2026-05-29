from src.data_generation.demo_scenarios import generate_demo_scenario
from src.reporting.data_dictionary import build_data_dictionary


def test_data_dictionary_documents_all_fields():
    frame = generate_demo_scenario("rds_stable_population", 200, 50, data_quality_level="none", random_seed=11)

    dictionary = build_data_dictionary(frame)

    assert len(dictionary) == len(frame.columns)
    assert {"champ", "libelle", "categorie", "type_technique", "description_metier", "taux_completude"}.issubset(dictionary.columns)
    assert "pd_estimate" in dictionary["champ"].tolist()
    assert dictionary.loc[dictionary["champ"] == "pd_estimate", "description_metier"].iloc[0]
