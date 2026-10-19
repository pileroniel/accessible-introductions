# Accessible Introductions

## Solving Accessibility Barriers for Visually Impaired People in Conversations

This project is an accessibility-focused computer-vision and machine-learning
application that generates descriptions of a person's appearance. Its purpose
is to provide visual context that sighted people often exchange naturally
during introductions and conversations, while reducing the need for a blind or
low-vision person to ask for that information directly.

The project is being developed as a CS 4900 senior project by Ron George Pile,
with Alex Wong as advisor.

## Problem

In a conversation, a brief visual impression can communicate information such
as clothing, colors, patterns, eyeglasses, hairstyle, and facial expression.
People with visual impairments may not have access to that context, and may
feel uncomfortable asking another person to describe themselves.

Visual impairment is a spectrum. The useful output may range from a shirt's
color for someone with color blindness to a fuller spoken description for
someone with little or no light perception. This project focuses on making
those descriptions available in an accessible, user-controlled form.

## Project goal

The long-term goal is a deployable application that can use a phone camera to
analyze people in a user's surroundings and provide concise, natural-language
appearance descriptions through text-to-speech or a screen reader.

The first-semester minimum viable product (MVP) intentionally limits the
problem to static, user-provided images:

```text
Image upload
    -> person detection and segmentation
    -> clothing-attribute classification
    -> structured attribute predictions
    -> template-based natural-language description
```

This staged approach allows the classification pipeline and its evaluation to
be developed before adding the complexity of live video and on-device
deployment.

## Planned MVP components

### Person detection and segmentation

A pretrained YOLO detection or segmentation model will identify people and
isolate relevant clothing regions. The project will use an existing pretrained
detector rather than building a person detector from scratch.

### Clothing-attribute classification

The primary area of original machine-learning work is a clothing-attribute
classifier. The classifier will use transfer learning with a pretrained CNN
backbone and the DeepFashion attribute-prediction dataset.

The work includes:

- selecting a useful subset of clothing attributes;
- collecting and preparing labeled data;
- preprocessing and augmentation;
- training and validation loops;
- baseline and transfer-learning comparisons;
- error analysis and hyperparameter tuning; and
- evaluation using accuracy, precision, recall, and confusion matrices.

### Natural-language generation

The MVP will convert structured predictions into readable descriptions using
templates. This keeps the output predictable and makes the model's behavior
easier to evaluate.

An LLM-based generation layer may be considered later, but it is not required
for the MVP.

### Web application and accessibility

The final MVP will provide a web application or progressive web application
(PWA) that accepts an image and returns a synthesized description. The
interface should be usable with accessibility tools such as screen readers
and should keep privacy, clarity, and user control central to the design.

## Code-first development

Jupyter notebooks are optional. The main application, training pipeline, and
evaluation should be implemented in regular Python modules and scripts so
that each step can be reproduced from a command line and tested independently.
Use notebooks only for exploratory analysis or temporary visualizations; do
not make the MVP depend on a notebook kernel.

From the project root, the intended workflow is:

```powershell
conda activate senior-project
python scripts/prepare_data.py
python scripts/train_classifier.py
python scripts/evaluate_model.py
python scripts/run_inference.py --image path\to\image.jpg
```

The first working vertical slice should be developed in this order:

1. Convert structured attributes into a concise, screen-reader-friendly
   template description.
2. Run pretrained YOLO on a static image and return person crops.
3. Prepare a documented subset of DeepFashion and implement its dataset
   loader.
4. Build the required non-deep-learning baseline classifier.
5. Fine-tune a pretrained CNN and compare it with the baseline.
6. Connect detection, classification, and template generation in the
   end-to-end inference pipeline.
7. Add the web/PWA upload interface around the inference pipeline.

Keep model training and data preparation out of the web application. The web
application should call the inference pipeline, while `scripts/` should
provide reproducible commands for preparation, training, and evaluation.

## Technology and data

- **Python**
- **PyTorch**
- **Ultralytics YOLO**
- **DeepFashion attribute-prediction dataset**
- **Convolutional neural networks and transfer learning**
- **Web application/PWA technology**
- **Optional text-to-speech or screen-reader output**
- **NVIDIA CUDA acceleration when supported by the development machine**

The DeepFashion dataset is intended for clothing and fashion-attribute
prediction. Dataset usage, redistribution, licensing, and storage should be
checked before sharing data or trained artifacts.

## Repository structure

The repository currently contains the YOLO detection proof of concept and the
directory layout for the remaining MVP components. Empty directories are kept
in place so new modules can be added without changing the project layout.

```text
Code/
├── README.md
├── .gitignore
├── architecture.png             # Current system architecture diagram
├── yolo26n.pt                   # YOLO model weights used by the trial
│
├── app/
│   ├── api/                     # Reserved for the upload/API service
│   └── web/                     # Reserved for the web/PWA interface
│
├── data/
│   ├── interim/                 # Reserved for intermediate preprocessing outputs
│   ├── processed/               # Reserved for model-ready datasets
│   └── raw/
│       └── ron_arsenal.jpg      # Current YOLO input image
│
├── docs/                        # Reserved for design, ethics, and data documentation
├── models/
│   ├── checkpoints/             # Reserved for training checkpoints
│   └── pretrained/              # Reserved for downloaded pretrained weights
│
├── notebooks/                   # Reserved for optional exploratory analysis
├── outputs/
│   ├── classifications/         # Reserved for classification results
│   └── detections/
│       └── ron_arsenal.jpg      # Annotated YOLO output
│
├── reports/                     # Reserved for metrics, plots, and summaries
├── scripts/                     # Reserved for repeatable CLI workflows
├── src/
│   ├── classification/          # Reserved for dataset and classifier code
│   ├── detection/
│   │   └── yolo_trial.py        # Current person-detection experiment
│   ├── evaluation/              # Reserved for metrics and error analysis
│   ├── generation/              # Reserved for description generation
│   └── inference/               # Reserved for the end-to-end pipeline
└── tests/                       # Reserved for unit and integration tests
```

The target MVP structure is the same layout with the reserved directories
populated by the application, training, inference, evaluation, and test
modules described below.

Large datasets, model weights, generated reports, and secrets should not be
committed to source control unless there is a specific reason to do so.

The main module responsibilities are:

| Module | Responsibility |
| --- | --- |
| `src/detection/` | Load the pretrained detector and produce person/clothing crops. |
| `src/classification/` | Load data, train the baseline and CNN, and predict attributes. |
| `src/generation/` | Convert structured predictions into predictable descriptions. |
| `src/inference/` | Orchestrate detection, classification, and generation for one image. |
| `src/evaluation/` | Calculate metrics, confusion matrices, and error-analysis reports. |
| `app/` | Accept an upload and present the pipeline result accessibly. |
| `scripts/` | Expose repeatable command-line entry points; do not contain reusable model logic. |

Recommended script entry points are:

```text
scripts/prepare_data.py       # raw DeepFashion files -> processed dataset
scripts/train_classifier.py   # processed dataset -> model checkpoint
scripts/evaluate_model.py     # checkpoint + held-out data -> reports/
scripts/run_inference.py      # image + checkpoint -> text description
```

## Development environments

The project uses a Conda environment for the main machine-learning application,
training, GPU inference, and the web pipeline:

```powershell
conda activate senior-project
```

Use `senior-project` for model training, GPU inference, and the application
pipeline.

No notebook environment is required to run the application or the command-line
scripts. If exploratory analysis is useful, it may be performed separately:

```powershell
conda activate ydata_fa26
jupyter notebook
```

Use `ydata_fa26` only for exploratory data analysis and notebooks. Confirm the
active interpreter before running project code:

```powershell
python -c "import sys; print(sys.executable)"
```

For GPU-enabled PyTorch, verify CUDA from inside `senior-project`:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU not detected')"
```

## Running without Jupyter

Run Python modules and scripts directly from the repository root:

```powershell
python scripts/run_inference.py --help
python -m pytest tests
```

Reusable code belongs under `src/`; command-line orchestration belongs under
`scripts/`; tests belong under `tests/`. This separation makes the same
pipeline usable from the terminal, the API, and later a phone or on-device
client.

## Evaluation

The classifier and end-to-end pipeline should be evaluated using:

- held-out test data;
- accuracy, precision, and recall;
- confusion matrices;
- error analysis;
- performance under different lighting and clothing conditions;
- inference latency where relevant; and
- usability and accessibility feedback when user testing is available.

Evaluation should distinguish between model quality and the usefulness of the
generated description. A correct attribute prediction is not automatically a
clear or helpful description.

## Ethics, privacy, and fairness

The application analyzes images of people, so privacy and informed use are
important design requirements. The project should:

- avoid retaining uploaded images unless retention is explicitly needed;
- document what data is processed and where it is processed;
- avoid presenting uncertain predictions as facts;
- evaluate performance across skin tones, clothing styles, and lighting
  conditions;
- avoid unnecessary or sensitive inferences about people; and
- include structured usability testing with blind and low-vision users when
  possible.

The application is intended to describe visible appearance attributes, not to
identify people or infer protected or sensitive characteristics.

## Semester-one milestones

1. Review related machine-learning, accessibility, and image-captioning work.
2. Set up the development environment and demonstrate pretrained YOLO on test
   images.
3. Acquire and preprocess an appropriate DeepFashion subset.
4. Build and evaluate a non-deep-learning baseline classifier.
5. Fine-tune a pretrained CNN using transfer learning.
6. Refine the model using error analysis, augmentation, and tuning.
7. Evaluate the held-out test set and prepare the report.
8. Build the web/PWA MVP and integrate the classifier and description module.
9. Complete the final report, poster, code, and data documentation.

## Future work

The second phase of the project may extend the MVP with:

- person re-identification and tracking;
- real-time video processing;
- direct phone-camera integration;
- on-device or edge inference;
- richer natural-language generation;
- an expanded but carefully reviewed attribute set;
- usability testing with blind and low-vision users; and
- broader bias and fairness evaluation.

## References

- Liu et al., “DeepFashion: Powering Robust Clothes Recognition and Retrieval
  with Rich Annotations,” CVPR 2016.
- Redmon et al., “You Only Look Once: Unified, Real-Time Object Detection,”
  CVPR 2016.
- Vinyals et al., “Show and Tell: A Neural Image Caption Generator,” CVPR
  2015.
- Xu et al., “Show, Attend and Tell: Neural Image Caption Generation with
  Visual Attention,” ICML 2015.
- Ye et al., “Deep Learning for Person Re-Identification: A Survey and
  Outlook,” IEEE TPAMI, 2022.
- [DeepFashion Attribute Prediction](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TorchReID](https://github.com/KaiyangZhou/deep-person-reid)
