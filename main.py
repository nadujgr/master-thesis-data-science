# main.py — pipeline entry point. Run from the project root with `python main.py`.

# Data preparation and preprocessing
from src.preparation.preprocessing import preprocessing_transcripts

# Cohort and transcript description
from src.description.metadata_analysis import run_metadata_analysis
from src.description.transcript_analysis import run_transcript_analysis

# Feature extraction and per-feature analysis
from src.features.topic_embeddings import embedding_generator
from src.features.lexical import run_lexical
from src.features.psycholinguistics import run_psycholinguistic
from src.features.speech_graphs import run_speech_graph_analysis
from src.features.cooccurrence_emotion import run_cooccurrence_nrc_analysis
from src.features.semantic_similarity import run_semantic_similarity
from src.features.semantic_similarity_network import run_semantic_network_analysis
from src.features.topic import run_topic_analysis

# Cross-patient analysis
from src.analysis.between_patient_stats import run_between_patient_stats
from src.analysis.classification import run_classification
from src.analysis.correlation_analysis import run_correlation
from src.analysis.longitudinal.longitudinal_analysis import run_longitudinal_analysis
from src.analysis.longitudinal.build_long_features import build_long_features

# Diagnostics
from src.diagnostics.verify_patient_data_consistency import verify_patient_data_consistency

def main():

    # print("Running data analysis...")
    run_metadata_analysis()
    # run_transcript_analysis()
    # print("Finished data analysis.")

    # print("Running topic embedding generation...")
    # embedding_generator(overwrite=True)  # set to True to re-generate all embeddings
    # print("Finished topic embedding generation.")

    # print("Running lexical analysis...")
    # run_lexical()
    # print("Finished lexical analysis.")

    # print("Running psycholinguistic analysis...")
    # run_psycholinguistic()
    # print("Finished psycholinguistic analysis.")

    # print("Running co-occurrence network analysis...")
    # run_cooccurrence_analysis()
    # print("Finished co-occurrence network analysis.")

    # print("Running speech graph analysis...")
    # run_speech_graph_analysis(metric_suffix="")     # raw metrics — primary analysis
    # run_speech_graph_analysis(metric_suffix="_z")   # structure-vs-random — robustness analysis
    # print("Finished speech graph analysis.")

    # print("Running co-occurrence analysis with emotion...")
    # run_cooccurrence_nrc_analysis()
    # print("Finished co-occurrence analysis with emotion.")

    # print("Running semantic similarity analysis...")
    # run_semantic_similarity(
    #     tasks=["emotion", "clinical_evaluation"],
    #     use_fasttext=True,
    #     use_sentence_transformers=True,
    #     use_xlm=True,
    #     run_stopword_experiment=False,  # saves semantic_patient_features.csv
    # )
    # run_semantic_similarity(
    #     tasks=["emotion", "clinical_evaluation"],
    #     use_fasttext=True,
    #     use_sentence_transformers=True,
    #     use_xlm=True,
    #     run_stopword_experiment=True,   # saves semantic_patient_features_stopword_exp.csv
    # )
    # print("Finished semantic similarity analysis.")

    # print("Running semantic network analysis...")
    # run_semantic_network_analysis(
    #     tasks=["emotion", "clinical_evaluation"],
    #     use_xlm=True,
    #     use_fasttext=True,
    #     use_sentence_transformers=False,
    #     run_stopword_experiment=False,  # saves semantic_networks_patient_features.csv
    # )
    # run_semantic_network_analysis(
    #     tasks=["emotion", "clinical_evaluation"],
    #     use_xlm=True,
    #     use_fasttext=True,
    #     use_sentence_transformers=False,
    #     run_stopword_experiment=True,   # saves semantic_networks_patient_features_stopword_exp.csv
    # )
    # print("Finished semantic network analysis.")

    # print("Running topic modeling analysis...")
    # run_topic_analysis() 
    # print("Finished topic modeling analysis.")

    # print("Running between-patient statistics...")
    # run_between_patient_stats()
    # print("Finished between-patient statistics.")

    # print("Running classification...")
    # run_classification()
    # print("Finished classification.")

  #   print("Running longitudinal analysis...")
  #   build_long_features()
  #   run_longitudinal_analysis()
  #   print("Finished longitudinal analysis.")

    # verify_patient_data_consistency()


if __name__ == "__main__":
    main()