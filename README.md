# Emotion Analyzer

A computer-vision project that estimates emotions from facial expressions using a CNN trained on FER2013, with a Streamlit dashboard and an AI agent that explains a recorded session in plain language, including the limits of the model.

> **Important:** the model estimates emotion from *facial expressions only* (about 56% test accuracy). It cannot know what a person thinks or feels, and it must not be used to judge people.

## Features

- **Emotion model:** MobileNetV2 fine-tuned on FER2013 (7 emotions), compared against a baseline CNN I trained from scratch
- **Live webcam recording (local):** face detection, emotion prediction and a session log (`webcam_app.py`)
- **Streamlit app:**
  - **Try it:** take a photo in the browser, see the predicted emotion and all 7 probabilities
  - **Session Report:** emotion distribution, timeline, segments, per-emotion confidence, with filters for confidence and short segments
  - **AI feedback:** an LLM agent that reads the session data through tools and writes a short, honest report
- **Interactive home page** showing how the pipeline works and how reliable each emotion is

## Results (FER2013 test set, 7,178 images)

| Model | Accuracy | Macro F1 | Disgust F1 |
|---|---|---|---|
| Baseline CNN (from scratch) | 53.9% | 0.435 | 0.00 |
| MobileNetV2 (transfer learning) | **56.1%** | **0.523** | **0.40** |

Per-emotion F1 of the MobileNetV2 model:

| angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|
| 0.46 | 0.40 | 0.35 | 0.78 | 0.51 | 0.44 | 0.71 |

What I learned from the evaluation:

- **Accuracy alone was misleading.** Accuracy only improved by about 2 points, but macro F1 went from 0.435 to 0.523 because the rare classes improved.
- **Class imbalance:** disgust has 436 training images versus 7,215 for happy. In the baseline, 77% of disgust images were predicted as angry and disgust was never predicted (F1 = 0).
- **Class weights needed tuning:** full "balanced" weights (disgust = 8.96) stopped training (loss stuck at 1.946, the value of a random 7-class guess). Square-root weights (disgust = 2.99) fixed it.
- **Hard classes:** fear, sad and angry are often confused with each other and with neutral. FER2013 labels are noisy, and more epochs did not help (the curves had flattened).

## How it works

```
camera -> face detection (Haar Cascade) -> crop, grayscale, 48x48 -> MobileNetV2 -> emotion + confidence
```

## The AI agent

The agent is an LLM, a set of Python tools and a loop. I wrote the loop myself, without an agent framework.

1. The LLM receives the question and a list of tool descriptions.
2. It asks for tools such as `get_session_overview`, `get_confidence_report`, `get_segments`.
3. My code runs the tools on the session data and returns the results.
4. This repeats until the LLM writes the final answer (with a maximum number of steps).

Design choices:

- **Honesty is built into the data, not only the prompt.** The tools return a `reliable` flag (at least 5 predictions and mean confidence >= 0.45) and the model's per-emotion test F1, so the agent knows which emotions to doubt.
- **Guardrails:** the system prompt forbids claims about feelings, lying, mental health or personality. I tested it with questions like "Was this person lying or nervous?", "Does this person have anxiety or depression?" and a leading question ("he was scared, right?"). In these tests it declined to infer feelings and pointed out the low-confidence predictions. LLM output is not deterministic, so this is a test, not a guarantee.
- **Provider-independent:** the loop only talks to a small adapter (`llm_gemini.py`), so another LLM can be added with one file.
- **Privacy:** only computed numbers are sent to the LLM, never photos.

## Project structure

```
app.py               Streamlit entry point (navigation and sidebar)
home.py              Home page
views/               Try it and Session Report pages
common.py            Shared constants and cached model loading
analytics.py         Session analysis (summary, segments, cleaning)
tools.py             Tools used by the agent
agent.py             Agent loop and system prompt
llm_gemini.py        Gemini adapter for the agent
webcam_app.py        Live webcam recording (local only)
data_loader.py       Data pipeline, class weights, augmentation
train_cnn.py         Baseline CNN training
evaluate.py          Test-set evaluation and confusion matrix
model/               Trained models and plots
```

## Setup

Requires **Python 3.12** (TensorFlow does not support 3.14 at the time of writing).

```bash
git clone https://github.com/<SakshiTikode>/emotion-analyzer.git
cd emotion-analyzer
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**Optional (AI feedback):** copy `.env.example` to `.env` and add a Gemini API key:

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.8-flash
```

Model names change over time. Pick one that your key can use.

**Optional (re-training or evaluating):** download FER2013 from Kaggle (`msambare/fer2013`) and place it as `data/train` and `data/test`.

## Usage

```bash
streamlit run app.py        # the full app
python webcam_app.py        # record a live session (press q to stop)
python evaluate.py best_mobilenet   # evaluate a model on the test set
```

Recorded sessions are saved under `sessions/` and can be opened on the **Session Report** page.

## Limitations

- Estimates expressions, not feelings. About 56% accuracy on FER2013.
- Real webcams differ from FER2013 images, so results can be worse in practice (dim light, odd angles).
- Disgust has very little data, and fear/sad/angry are often confused. The test set has only 111 disgust images, so its score is noisy.
- Live webcam recording works only locally. A deployed app has to use browser camera photos.

## Privacy and ethics

- Photos are processed in memory and are not saved. Only the predicted emotion is kept for the current browser session.
- Emotion recognition is sensitive and is restricted in some places (for example in workplaces and schools under the EU AI Act). This is a learning project and should not be used to make decisions about people.

## Future work

- Oversampling or focal loss for rare classes; the cleaner FER+ labels
- Ensemble of the two models and test-time augmentation
- Live video in the browser (WebRTC)
- A chat box where users ask the agent questions about a session