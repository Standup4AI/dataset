# StandUp4AI: A Multilingual Dataset for Humor Detection in Stand-up Comedy

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the code and dataset for our EMNLP 2025 submission: **"StandUp4AI: A New Multilingual Dataset for Humor Detection in Stand-up Comedy Videos"**.

## 📊 Dataset Overview

StandUp4AI is the largest and most linguistically diverse multilingual dataset of live comedy performances, containing:
- **3,617 stand-up comedy videos** in **7 languages**
- **334 hours** of content
- **~3 million words** transcribed
- **130,000+ laughter labels**

### Languages Covered
- 🇬🇧 English (Comedy Central US/UK)
- 🇫🇷 French (Montreux Comedy)
- 🇪🇸 Spanish (Comedy Central Latam/España)
- 🇮🇹 Italian (Comedy Central Italia)
- 🇵🇹 Portuguese (Comedy Central Brasil)
- 🇭🇺 Hungarian (Comedy Central Magyarország)
- 🇨🇿 Czech (Paramount Network CZ)

## 🚀 Key Features

1. **Sequence Labeling Approach**: Unlike traditional binary classification, we model humor detection as a word-level sequence labeling task
2. **ASR-Enhanced Laughter Detection**: Novel method combining Whisper and WhisperX outputs to fix timestamp errors and detect additional laughters
3. **Multimodal Annotations**: Includes transcripts, laughter timestamps, and extracted features (action units, poses, camera angles)
4. **Manually Annotated Test Set**: 70 videos (10 per language) with precise laughter annotations

## 📁 Repository Structure

```
standup_comedy/
├── URL_videos.ipynb              # Extract video URLs from YouTube channels
├── descarga.py                   # Video downloader using yt-dlp
├── normalize_names_id.py         # Standardize video filenames
├── src/
│   ├── extract_audio.py          # Extract audio from videos
│   └── asr_pipeline.py           # ASR transcription pipeline
├── ASR/
│   ├── whisperX.py               # WhisperX implementation
│   ├── utils.py                  # ASR utilities and timestamp fixing
│   └── create_validation_videos.py
├── labeling/
│   ├── training.py               # Sequence labeling model training
│   └── create_seq_labels.py      # Generate sequence labels
├── Examples_label/               # Example labeled CSV files
└── CSV_clean/                    # Channel URL lists
```

## 🛠️ Installation

### Requirements
- Python 3.8+
- Pytorch
- [WhisperX](https://github.com/m-bain/whisperX/tree/main) 
- CUDA-capable GPU (recommended)
- ffmpeg
- transformers 
- [Laughter Segmentation from Omine et al. (2024)](https://github.com/omine-me/LaughterSegmentation): pydub, librosa

### Setup
```bash
# Clone the repository
git clone https://github.com/[username]/standup_comedy.git
cd standup_comedy

# Install dependencies
pip install -r requirements.txt

# Install yt-dlp (for video downloading)
python3 -m pip install -U "yt-dlp[default]"
```

## 📥 Data Download and Processing Pipeline

### 1. Extract Video URLs from YouTube Channels
```python
# Use the URL_videos.py to extract stand-up comedy video URLs
# This filters videos by title (e.g., "| Stand Up |", "#StandupEnComedy")
# and excludes short videos (< 60 seconds)

# Example usage:
from URL_videos import get_standup_video_urls

channel_url = "https://www.youtube.com/@ComedyCentralLA/search?query=%22%7C%20Stand%20Up%20%7C%22"
df = get_standup_video_urls(channel_search_url, list_search_str=["| Stand Up |", "#StandupEnComedy",
                                                              "| CC Emergente |", "| COMPLETO |",
                                                              ])
df[df.duration > 60].to_csv('ComedyCentralLA_standup.csv', index=False)
```

### 2. Download Videos
```bash
# Download videos with subtitles using the CSV file from step 1
python descarga.py <language_code> <urls_file.csv>

# Example:
python descarga.py es Spanish_URL_standup.csv
```

### 3. Extract Audio
```bash
python src/extract_audio.py -i /path/to/videos -o /path/to/audio
```

### 4. Run ASR Pipeline
```bash
# Using Whisper
python src/asr_pipeline.py \
    --input_dir /path/to/audio \
    --output_dir /path/to/transcripts \
    --language <lang_code> \
    --model_id openai/whisper-large-v3
```

```python
# Using WhisperX (for word-level timestamps)
from whisperX import whisperX_main

lang='fr'

list_files_index = [os.path.split(k)[1].split('.')[0] for k in glob.glob(f'path/to/wav/{lang}/' + "*.wav")]

whisperX_main(list_files_index = list_files_index, # list of indexes
            path_wav = path_wav,
            path_csv = path_data_standup + f'transcript/{lang}/whisper_model_name/',
            model_name = "path/to/faster_whisper_model", # can use "large-v3", or any model converted previously to fast-whisper format, see https://github.com/SYSTRAN/faster-whisper#model-conversion
            device = "cuda",
            batch_size = 32, # reduce if low on GPU mem
            compute_type = "float16", # change to "int8" if low on GPU mem (may reduce accuracy)
                )
```

### 5. Extract Laughters

#### 5.1 Using base off-the-shelf model
Using the model from [Omine et al. (2024)](https://github.com/omine-me/LaughterSegmentation), we extract automatically laughters with the `inference_val.py` file. 

#### 5.2 Create potential candidates laughters using ASR

Using WhisperX and Whisper as inputs, the `ASR/create_validation_videos.py` file creates a csv with the list of candidates.  


#### 5.3 Validate the candidates 

After a manual annotation of 50 videos regarding **if candidate laughters were real laughters or not**, a RF model has been trained on the audio to automatically discriminate between a real laughter and a something that is not, creating the `filtered_candidates` predictions.


#### 5.4 Validate predictions of ASR-based and off-the-shelf methods

On a set of 70 videos manually annotated in laughter, we compare the performances of the initial model and the model using filtered candidates.  

```python
from validation_predictions import validation_test_set, calculate_metrics

# filtered_candidates, all_candidates, initial_laughters

for type_pred in ['initial_laughters','filtered_candidates']:
    TP_all, FP_all, FN_all = 0, 0, 0
    for lang in ['fr', 'es', 'it', "cs", 'hu', 'en_uk', "pt",
                ]:
        TP, FP, FN = validation_test_set(iou_threshold=0.2, 
                            path_pred=f'/path/to/laughters/after/prediction/{lang}/',
                            path_gt=f'/path/to/annotated/laughters/{lang}/',
                            type_pred = type_pred, verbose=False)
        TP_all += TP
        FP_all += FP
        FN_all += FN

    precision, recall, accuracy, f1= calculate_metrics(TP_all, FP_all, FN_all)   
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1: {f1:.2f}")
    print('\n\n')
```


### 6. Generate Laughter Labels
```python
from create_seq_labels import create_files_corpus

for lang in ['en_uk', 'fr', 'es', 'cs', 'hu', 'it', 'pt']:
    # create files that have been manually annotated
    create_files_corpus(lang=lang, Manual_annot=True)

```

Examples of automatically labeled data are available in the Examples_label [folder](./Examples_label). 

## 🤖 Training Humor Detection Models

### Sequence Labeling Model

To train a model on the filtered data 
```bash
python labeling/training.py \
    --lang fr \
    --train_folder Filtered \
    --dev_folder Filtered \
    --nb_epoch 10 \
    --lr 1e-5 \
    --save_model \
    --evaluate_val \
    --test
```


### Evaluation
```bash
python labeling/training.py \
    --lang <language_code> \
    --only_test \
    --test
```

## 📊 Results

### Laughter Detection Performance
- **Off-the-shelf model**: F1 = 0.51
- **Our ASR-enhanced method**: F1 = 0.58

### Humor Detection (Sequence Labeling)
| Model Type | Average F1 |
|------------|------------|
| Multilingual (Raw) | 42.2 |
| Multilingual (Enhanced) | 42.4 |
| Monolingual (Enhanced) | 39.4 |

## 📄 Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{standup4ai2025,
  title={StandUp4AI: A New Multilingual Dataset for Humor Detection in Stand-up Comedy Videos},
  author={Anonymous},
  booktitle={EMNLP 2025 Submission},
  year={2025}
}
```

## 🔗 Links

- **Paper**: EMNLP 2025 Submission
- **Dataset**: Available on demand (research purposes only)


## ⚠️ Ethical Considerations

- This dataset contains comedy content that may include adult themes and language
- Videos are sourced from public YouTube channels
- We only distribute metadata and annotations, not the actual video/audio content
- Please respect copyright and use this dataset for research purposes only

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the authors through the EMNLP submission system.

---

**Note**: This repository is under active development. Features and documentation may be updated as we prepare for the final EMNLP 2025 submission.
