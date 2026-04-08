# Semantic Similarity Linking - Danish National Archives
Python-based SBERT pipeline for entity matching of historical records using semantic similarity.

This project was developed as part of work with the Danish National Archives, focusing on linking individuals across historical census datasets with ambiguous, incomplete, and inconsistently formatted records.

## Overview
The pipeline uses Sentence-BERT (SBERT) embeddings to compare textual record fields and identify likely matches between entities across datasets.

Main components include:

- Data preprocessing and text normalization
- Semantic embedding generation with SBERT
- Similarity-based entity matching
- Model evaluation against ground-truth matches

## Data
The historical census data used in this project can be downloaded from the Danish National Archives:
https://digidata.rigsarkivet.dk/aflevering/14001
