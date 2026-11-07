# Predicting Airbnb price levels in Melbourne from photos, text and listing details

Project 2.2 for Techniques for Multimodal Data Analysis, Adelaide University, 2026.
Abu Wakkas Bastob

## About the project

The goal is to sort Melbourne Airbnb listings into three price levels: budget (up to $200 a night), mid-range ($200 to $323) and premium (above $323). Each listing has three kinds of data: its main photo, the host's description, and the listing details such as room type, number of guests, bedrooms, amenities, reviews and area.

I trained a model on each type of data alone, then compared four ways of combining them: early fusion, late fusion, concatenation fusion and attention fusion. After that I tested the models on listings from hosts they had never seen, and checked robustness, uncertainty, fairness across areas and how long everything takes to run.

Main result: the best fusion model (attention fusion, macro F1 0.671) only matched the best single model (the tabular MLP, 0.669). A bootstrap test shows the difference is just noise. Most of the useful information is in the listing details, and the photo adds very little.

## What's in this repo

- notebooks: the five notebooks, numbered in the order they run
  - 01_photos_download: gets the listings file and downloads one photo per listing
  - 02_data_preparation: groups related listings, makes the train/validation/test split, sets the price levels and cleans the descriptions
  - 03_extract_features: photo features (EfficientNet-B0, ResNet-50), text features (BERT) and photo quality checks
  - 04_models: single-modality baselines, the fusion models, tuning and 3 seeds (validation set only)
  - 05_evaluation: everything on the test set, including the bootstrap, robustness, CKA, MC dropout, fairness and efficiency
- results: all the tables used in the report, plus loss_curves.png from notebook 04
- results/figures: the test set figures from notebook 05
- requirements.txt: the package versions I used

The notebooks still have their outputs, so you can see all the results without running anything. The only outputs I removed are two grids of listing photos (in notebooks 01 and 03), because the photos belong to the hosts. The code for them is still there, so they come back if you run it.

What I left out on purpose:

- The raw data, photos, extracted features and model weights. They're too big, and I said in the proposal that the photos would only be used as model input, not shared.
- Any host information. Notebook 02 drops all the host columns and the exact locations, and removes names, phone numbers and emails from the descriptions.

## Data

Inside Airbnb, Melbourne, snapshot from 16 September 2026 (25,715 listings). Licence: CC BY 4.0.
https://insideairbnb.com/get-the-data/

You don't need to download anything by hand, because notebook 01 gets the listings file itself. If Inside Airbnb has moved this snapshot to their archive by the time you run it, you can request it from them and put it in airbnb_p22/data/raw/ with the name listings_2026-09-16.csv.gz.

## How to run it

I ran everything on Google Colab. Each notebook connects to Google Drive and saves its files in a folder called airbnb_p22, so the next notebook can carry on from there. You'll need about 2 GB free on Drive.

Run the notebooks in order (Runtime, then Run all):

1. 01_photos_download, on CPU. Takes about 80 to 90 minutes because it downloads 19,004 photos.
2. 02_data_preparation, on CPU. A few minutes.
3. 03_extract_features, on a T4 GPU. About 10 minutes.
4. 04_models, on a T4 GPU. A few minutes.
5. 05_evaluation, on a T4 GPU. About 6 minutes.

If you're not using Colab, install the packages first:

    pip install -r requirements.txt

The notebooks then use a local airbnb_p22 folder instead of Drive. Skip the last cell of each notebook, because it only saves to Google Drive.

## Reproducibility

- Every random step has a fixed seed: the split uses 42, and the neural networks use 42, 43 and 44.
- Every setting was chosen on the validation set in notebook 04. The test set is only used once, in notebook 05.
- The price cut-offs, the price cap, the scaling and the TF-IDF vocabulary all come from the training set only.
- The split keeps each host's listings together, so no host appears in both training and test.
- Photo links stop working over time. When I downloaded them (4 to 6 November 2026), 18,802 of 19,004 still worked. A later run might get a few fewer photos, which could shift the image results slightly. Everything else comes from the fixed snapshot.

## Test results (macro F1)

Neural models are the average of 3 seeds.

- LR tabular: 0.664
- LR text (TF-IDF): 0.596
- LR image (EfficientNet-B0): 0.478
- Gradient boosting tabular: 0.661
- MLP tabular: 0.669
- Early fusion (LR): 0.654
- Late fusion (weighted): 0.671
- Concat fusion: 0.669
- Attention fusion: 0.671

The full tables are in the results folder and explained in the report.

## Pretrained models

- EfficientNet-B0 and ResNet-50 from torchvision, with ImageNet weights
- bert-base-uncased from Hugging Face, with mean pooling
