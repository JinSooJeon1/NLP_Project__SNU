## Personality Prediction from Text ## 
Unified Big Five and MBTI Multitask NLP Pipeline

This repository provides a complete data preprocessing pipeline and a psychology-informed multitask model for predicting Big Five and MBTI personality traits from text. The system integrates multiple public and private datasets, normalizes labels into consistent formats, and optionally applies emotion-based filtering to emphasize psychologically meaningful signals.

Some preprocessing functions and structural ideas were referenced from the open-source project at:
https://github.com/jkwieser/personality-prediction-from-text

Modifications include major structural reorganization, complete support for HuggingFace datasets, MBTI label preservation, unified Big Five schema, and a redesigned multitask architecture.


## Key Features ## 

Supported data sources include:

HuggingFace dataset "jingjietan/pandora-big5"
Local file "essays.csv"

Both datasets are converted into the unified schema:
TEXT, cEXT, cNEU, cAGR, cCON, cOPN
Outputs are stored as Essay objects (essay.Essay).

MBTI Dataset Builder
Supported data sources include:

Kaggle MBTI dataset "mbti_1.csv"

Emotion Lexicon Filtering (optional)
If Emotion_Lexicon.csv is provided, the pipeline can remove non-emotional sentences or give preference to emotionally rich sentences.
This filter applies only to Big Five text (essay datasets).
MBTI text is kept unfiltered to preserve type-specific writing patterns.

## Output Files ##
Big Five output files example:
essays_big5_<N>.p
essays_big5.p

MBTI output files example:
text_mbti_<N>.p
text_mbti.p



## Model Architecture ##

The model jointly predicts the Big Five traits (five binary outputs) and MBTI dimensions (four binary two-class outputs). A psychology-based trait correlation graph connects Big Five traits with MBTI dimensions.

Dual Encoders
Because Big Five traits and MBTI axes reflect different linguistic tendencies, the architecture uses two separate encoders:

A Big Five encoder that outputs a 5-dimensional trait prototype

An MBTI encoder that outputs a 4-dimensional trait prototype

Trait Correlation Graph
The model includes a graph with 9 nodes (5 Big Five traits plus 4 MBTI axes).
Edges are initialized using correlations reported in personality psychology research.
Message passing strengthens relationships such as:
Extraversion (Big Five) corresponding to MBTI E/I,
Openness corresponding to MBTI N/S, etc.

Multitask Output Heads
The model predicts:

5 binary labels for Big Five

4 binary labels for MBTI

Loss masking is used depending on dataset type:
Big Five datasets mask MBTI loss,
MBTI datasets mask Big Five loss.



## Usage ##

Run preprocessing
Run preprocessing.py to generate all processed datasets.
Big Five and MBTI outputs will be saved under data/essays/.

Train the multitask model
Use the scripts in the models directory to initialize, train, and evaluate the PersonalityModel.
