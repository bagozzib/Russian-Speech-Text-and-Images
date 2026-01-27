# Linked Multi-Model Data on Russian Domestic and Foreign Policy Speeches

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [File Structure and Usage](#folder-structure-and-usage)
- [Contributors](#contributors)
- [License](#license)


## Description:
This repository builds a linked, multimodal dataset of Russian political speeches and related images from two official sources — the Kremlin (President of Russia) website and the Russian Ministry of Foreign Affairs (MID.ru) — across English and Russian corpora. The pipeline scrapes speech pages, extracts and cleans metadata (IDs, URLs, dates, titles, locations/lat/long, speakers, full text and English translations where applicable), and downloads associated page images into per-speech folders. For topic modeling, it runs BERTopic (SentenceTransformers + UMAP + HDBSCAN) to generate top-K topic candidates per speech and then assigns TOP-1 curated topics using manually curated topic dictionaries (topic_id → label + group) for each corpus.

In addition, the repo includes image-topic scoring that assigns a TOP-1 topic per image using CLIP-based similarity scoring, exporting an image-topics file aligned to each speech’s stored_image_filepaths order. Finally, the pipeline merges all outputs back into a single master CSV by adding standardized curated columns for both text and images: curated_topic_id, curated_text_topic_label, curated_text_topic_group, curated_topic_probability, and the aligned per-image arrays curated_image_topic_ids, curated_image_topic_labels, curated_image_group_names, curated_image_topic_probabilities.

## Installation:
To set up the project environment, follow these steps:

1. **Clone the repository:**
   ```
   [git clone https://github.com/bagozzib/CITES_Data.git](https://github.com/bagozzib/Russian-Speech-Text-and-Images.git)
   ```
   
2. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

## File Structure and Usage:

### `python_code/` (core pipeline)


- **`prepare_index/`**  
  Small helper scripts to prepare consistent **ID lists / index inputs** used downstream.
  - `extract_ids_kremlin.py` — builds/updates Kremlin ID/index inputs  
  - `extract_ids_mid.py` — builds/updates MID ID/index inputs
    
- **`webscraping_code/`**  
  Scrapers that collect **speech pages, metadata, and images** from each source site.
  - `kremlin/` — Kremlin-specific scraping, and parsing logic  
  - `MID/` — MID-specific scraping, and parsing logic  

- **`bertopic_text_image_pipeline_all_datasets.ipynb`**  
- The notebook runs **BERTopic** on the speech text (for **RU corpora it models `full_text_english`**, so topics align with EN).
- It assigns **TOP-1 topic per speech** and produces topic metadata (e.g., keywords/sizes).
- It then uses **CLIP-based scoring** to compare each image to topic representations and assigns each image a **TOP-1 topic** (highest similarity score).
- Saves the updated CSVs/outputs for downstream analysis and visualization.
  
- **`Validation/`**  
  Sanity checks to validate key outputs before publishing.
  - `topic_validation.py` — checks topic outputs (missing topics, row counts, basic consistency)  
  - `latitude_longitude_validation.py` — checks location fields / coordinate sanity

---

## Combined usage (recommended run order)

1) **Prepare IDs / index inputs**  
   Run `python_code/prepare_index/extract_ids_kremlin.py` and `python_code/prepare_index/extract_ids_mid.py` so downstream steps reference stable, consistent speech IDs.

2) **Scrape datasets (texts + images)**  
   Run the scrapers inside `python_code/webscraping_code/` for **Kremlin** and **MID** to generate the extracted speech CSVs and download images.

3) **Topic modeling + image-topic scoring**  
   Open and run:  
   `python_code/bertopic_text_image_pipeline_all_datasets.ipynb`  
   This produces topic assignments for speeches (BERTopic) and images (CLIP scoring).

4) **Validate outputs**  
   Run scripts in `python_code/Validation/` to confirm outputs are clean (topic coverage, missingness, coordinate sanity where applicable).

## Conclusion
- We built a reproducible end-to-end pipeline that collects, cleans, and harmonizes Russian official speech texts and associated webpage images across two sources (Kremlin and MID) and four corpora (EN/RU per source).
- The workflow standardizes metadata and IDs, models topics consistently using BERTopic (with the RU corpora modeled on `full_text_english` to align topic space with English), and assigns each image a TOP-1 topic via CLIP-based image–topic similarity scoring.
- The result is a unified, analysis-ready dataset linking **speeches ↔ topics ↔ images ↔ image-topics**, with lightweight validation scripts to sanity-check topics and geolocation fields.

## Contributors
- Benjamin E. Bagozzi (Corresponding author: bagozzib@udel.edu)
- Sunita Chandrasekaran
- Daria Blinova
- Gayathri Emuru
- Rakesh Emuru
- Kushagradheer Shridheer Srivastava
- Mina Rulis
     
## License

#### This project is licensed under the Creative Commons Attribution 4.0 International License (CC-BY-4.0).
   - This license allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator. The license allows for commercial use.

For more details about this license, please visit the [Creative Commons Attribution 4.0 International License webpage](https://creativecommons.org/licenses/by/4.0/).



   
