# UA Locations Extraction: Recurrent NER (BiLSTM)

Solution for the Kaggle in-class competition **"UA Locations Extractions"**: find all location mentions (cities, regions, countries, streets, addresses, landmarks) in Ukrainian, Russian and mixed-language texts.

The task is solved as **token-level Named Entity Recognition** with a **bidirectional LSTM tagger**, built with **PyTorch + PyTorch Lightning** and tracked with **Weights & Biases**.

**Constraints of the assignment:** recurrent models only (LSTM/GRU), no transformers or attention, no Keras/TensorFlow.

| | Validation F1 (competition metric) | `labeling_sample` F1 | Public LB |
|---|---|---|---|
| BiLSTM baseline (ours) | 0.910 | 0.548 | **TODO** |
| **Best: BiLSTM + fastText init** | **0.915** | 0.615 | **TODO** |
| Organizers' spaCy baseline | – | – | **TODO** |

W&B project: <https://wandb.ai/lyaroshenko-kse/ua-locations-ner>

---

## 1. Data insights

### Files

| File | Rows | Content |
|---|---|---|
| `uk_geo_dataset.csv` | 1,010,000 | Ukrainian sentences (Ukr-Synth) with character spans for LOC / ORG / PER, `is_valid` flag |
| `ru_geo_dataset.csv` | 8,028,840 | Russian sentences (Nerus) with LOC / ORG / PER spans, `doc_id`, `sent_id` |
| `labeling_sample.csv` | 26 | Texts from the **test distribution** with gold locations |
| `test.csv` | 477 | Texts to predict (the `locations` column is an empty placeholder) |

**Label format.** Labels are **character offsets** `(start, end)` with an exclusive end, stored as strings and parsed with `ast.literal_eval`. For example, `"Проспект Бандери"` is `(0, 16)`, and `text[0:16]` gives exactly that string. There are no token tags in the data; we build them ourselves.

**Submission format.** For every `text_id`, a JSON list of location strings **in the exact surface form found in the text**, for example `["Києва", "Хелма"]`.

### Exploratory statistics

| Dataset | n | Median length (chars) | Max length | Has `\n` | Texts with 0 locations | Mean locations / text |
|---|---|---|---|---|---|---|
| uk | 1,010,000 | 91 | 255 | 0.0% | 76.9% | 0.34 |
| ru (first 300k rows) | 300,000 | 100 | 762 | 0.0% | 72.0% | 0.43 |
| labeling_sample | 26 | 218 | 1,118 | 80.8% | 38.5% | 1.23 |
| test | 477 | 264 | 3,682 | 85.1% | – | – |

<!-- Optional: add the EDA figure, e.g. ![EDA](images/eda.png) -->

### Key observations

1. **Strong domain shift.** Training data are clean, single news sentences. Test data are long, multi-line Telegram posts with emojis: about 3× longer at the median, up to 3.7k characters, and 85% contain line breaks.
2. **Mixed language is real.** `labeling_sample` contains Russian word forms such as "Фастовском" and "Бучанском" inside Ukrainian posts, so we train on Ukrainian **and** Russian data.
3. **Locations are inflected.** "Києва", "Києву", "Києвом" and "Києві" all appear, and the metric does no lemmatization, so the model must return the exact form from the text.
4. **Different labeling conventions.** The test-like sample labels streets and addresses (`"проспекті Івасюка"`, `"вул. Солом'янська, 28"`), districts (`"Шевченківському районі"`) and even a shopping mall (`"ТЦ Аркадія"`). The training corpora mostly label cities, regions and countries. This limits recall on the test set.
5. **Class imbalance.** 72–77% of training sentences contain no location, and at the token level "O" dominates even more. Plain cross-entropy turned out to be sufficient.
6. **Characters.** Only 2.0% of test characters (mostly emojis) never appear in the Ukrainian training text.

---

## 2. Evaluation metric

### How it works

For each text, both the gold and the predicted location strings are normalized (`hard_processing`: lowercase, remove digits and one-letter words, collapse spaces) and compared as **sets**. True positives, false positives and false negatives are **summed over the whole test set** (micro-averaging), and a single precision, recall and F1 are computed from these totals. Only strings matter, not their positions in the text.

### Hidden detail: punctuation is not removed

The punctuation-removal line in the official code is missing the `f` prefix:

```python
re.sub(r"[{re.escape(string.punctuation)}]", "", text)   # not an f-string
```

So the regex is the literal character class `[{re.escape(string.punctuation)}]`. It removes only `. ( ) { }` and a few Latin letters. **Commas, apostrophes, hyphens and quotes stay.** We checked this directly:

| Input | After `hard_processing` |
|---|---|
| `"вул. Солом'янська, 28"` | `"вул солом'янська,"` |
| `"Київ,"` | `"київ,"` (does **not** match `"київ"`) |
| `"м. Київ"` | `"київ"` |

Because of this, we trim leading and trailing punctuation from predicted strings (see §4). The local metric is copied verbatim (including the missing `f`) so that local scores match Kaggle.

### Strengths
1. **Independent of tokenization.** Only final strings are compared, so character, word and subword models are scored identically.
2. **Fair for texts without locations.** Micro-averaging means empty texts neither help nor hurt, and both missed and invented locations are penalized.

### Weaknesses
1. **All or nothing.** Predicting `"Шевченківському"` against the gold `"Шевченківському районі"` gives zero credit and is counted twice (1 FP + 1 FN).
2. **Texts with many locations dominate, and the test set is small.** One post listing 10 towns weighs as much as 10 short posts, and with only 477 test texts the score is noisy. The punctuation bug adds noise unrelated to model quality.

### Edge cases
1. **Wrong word form:** `"Київ"` against the gold `"Києва"` scores 0, because there is no lemmatization.
2. **Empty predictions and duplicates:** predicting `[]` for a text without locations is neutral. Duplicates in one text collapse into one (sets). A string that becomes empty after normalization (e.g. `"28"` becomes `""`) is still a set element and can produce a false positive.

### Complementary metrics
1. **Precision and recall reported separately,** to show whether the model misses locations or invents them.
2. **Token-level or character-level F1** on the large validation split. It gives partial credit for overlapping spans and is far less noisy than 26 texts.

### Alternative target metric
**Partial-match entity F1** (as in the "partial" mode of the SemEval-2013 NER evaluation): an overlapping prediction earns half credit, or credit proportional to word overlap. For downstream uses like geocoding or search, a nearly correct span is far more useful than a miss, and the metric is more stable on small evaluation sets.

---

## 3. Validation strategy

- **Training pool:** all 1,010,000 Ukrainian sentences + the first 500,000 Russian rows, 1.51M sentences in total.
- **One fixed 80/20 split (seed 42)**, saved as indices (`split.npz`) and reused by every experiment.
- **Grouped split for Russian:** sentences from the same article (`doc_id`) never end up on both sides, which would leak names and places into validation. Ukrainian sentences have no group or time column and are split randomly (each sentence is its own group). One `GroupShuffleSplit` call handles both.
- **Per-epoch validation** uses a fixed random subset of 20,000 validation sentences, because scoring all 300k every epoch is too slow.
- **Test-like check:** `labeling_sample` (26 texts) is scored with the competition metric for every model, but it is not used for early stopping because it is too small.

**Why validation may not match the leaderboard.** The validation set comes from the same clean news corpora as training, while the test set consists of Telegram posts with a different labeling convention. Validation scores will therefore be much higher than leaderboard scores and may rank models differently. Our experiments confirm this (see §7).

---

## 4. Feature engineering and preprocessing

| Step | What we do | Why |
|---|---|---|
| **Tokenization** | Regex `\w+(?:['’ʼ-]\w+)*\|[^\w\s]`, keeping each token's character start and end | Keeps Ukrainian apostrophe and hyphenated words whole ("Солом'янська", "Київ-Пасажирський"). The offsets let us return exact substrings. |
| **Labels** | Character spans become **BIO** tags (`O`, `B-LOC`, `I-LOC`). A token is inside a span if it **overlaps** it. | Standard NER scheme. Overlap keeps tokens whose span boundary falls mid-token. Round-trip check: spans → tags → spans gives back the original spans. |
| **Vocabulary** | Words with frequency ≥ 2 in the **train split only**, **case preserved**, plus `<pad>` = 0 and `<unk>` = 1 | Capitalization is a strong signal for proper names. Building on train only keeps the `<unk>` rate on validation realistic. |
| **Batching** | Pad to the longest sequence in each batch. Tags padded with `-100` (ignored by the loss). Boolean mask returned. | Padding never affects the loss. |
| **Packed sequences** | `pack_padded_sequence` in the forward pass | The RNN never reads padding. This matters for the backward direction of the BiRNN. |
| **Inference: line splitting** | Each test text is tokenized as a whole, then the token list is cut wherever there is a `\n` between tokens. Each line is tagged separately. | Each line resembles a training sentence. Offsets still refer to the full text. |
| **Decoding** | Tags become spans in **strict** mode (B starts an entity, I continues it, an I without a preceding B is dropped) or **loose** mode (any run of non-O tokens is one entity). Strings are cut from the text, leading and trailing punctuation is trimmed, and duplicates are removed. | Strict was better or equal on `labeling_sample` for every model. Trimming works around the metric bug. |
| **Pretrained embeddings (exp. e)** | fastText `cc.uk.300` vectors loaded only for vocabulary words (exact form first, then lowercase); random init for the rest; fine-tuned | 77.1% of the vocabulary was covered. |
| **Subword tokens (exp. f)** | HuggingFace `tokenizers` BPE (tokenizer only), vocabulary of 30k, `Whitespace` pre-tokenizer; labels mapped to subwords by the same overlap rule | Fewer unknown words and a 12× smaller embedding table. |

---

## 5. Model

```
token ids ─► Embedding (d=128) ─► 2-layer BiLSTM (hidden 256 per direction, dropout 0.3 between layers)
          ─► Dropout(0.3) ─► Linear(512 → 3) ─► softmax over {O, B-LOC, I-LOC}
```

- **Loss:** cross-entropy over tokens with `ignore_index=-100` for padding.
- **Size (baseline):** 50.4M parameters, of which **48.0M are the embedding table** (~375k words × 128), 2.4M the LSTM and 1.5K the output layer.
- **Configurable:** RNN type (LSTM/GRU), number of layers, bidirectionality, optimizer (Adam or AdamW + OneCycleLR), optional pretrained embedding matrix. Everything is set from one `config` dict, so each experiment changes exactly one value.
- **Logged to W&B every epoch:** `train_loss`, `val_loss`, `val_f1` (competition metric). Every step: per-layer and total **gradient L2 norms** (before clipping) and, for OneCycle, the learning rate.

---

## 6. Training setup and chosen hyperparameters

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 128 (300 for fastText) |
| RNN | LSTM, 2 layers, bidirectional, hidden size 256 |
| Dropout | 0.3 (between RNN layers and before the output layer) |
| Optimizer | Adam, lr = 1e-3 |
| Batch size | 256 (~4,717 steps per epoch) |
| Gradient clipping | L2 norm 1.0 |
| Max epochs / early stopping | 10 / patience 2 on `val_f1` |
| Checkpointing | best `val_f1` only |
| Vocabulary | min_freq = 2, case-sensitive, ~375k words |
| Seed | 42 |
| Hardware | Colab, NVIDIA L4, ~5 min per epoch |

These are reasonable defaults for a BiLSTM tagger. We did not run a large search; instead we ran controlled one-change ablations (§7). Training stops early at around epoch 5, so the total cost is about 25–30 minutes per run.

---

## 7. Experiments

Each experiment changes **one** setting from our BiLSTM baseline. Everything else (data, split, seed, vocabulary, decoding) is identical.

| Experiment | Change | Val F1 | `labeling_sample` F1 (strict / loose) | Best epoch | Public LB |
|---|---|---|---|---|---|
| **BiLSTM baseline** | – | 0.9099 | 0.548 / 0.548 | 3 | **TODO** |
| (a) GRU | LSTM → GRU | 0.9047 | 0.615 / 0.593 | 2 | **TODO** |
| (b) 1 layer | 2 → 1 RNN layer (between-layer dropout off) | 0.9001 | **0.657** / 0.622 | 2 | **TODO** |
| (c) Unidirectional | BiLSTM → forward LSTM | 0.8898 | 0.583 / 0.575 | 3 | **TODO** |
| (d) AdamW + OneCycleLR | weight decay 0.01, peak lr 3e-3, early stopping off so the cycle completes | 0.9100 | 0.613 / 0.605 | 4 | **TODO** |
| **(e) fastText init** | embeddings initialized from `cc.uk.300` (dim 300, 77.1% coverage) | **0.9152** | 0.615 / 0.615 | **1** | **TODO** |
| (f) BPE subwords | word tokens → 30k BPE subwords | 0.9072 | 0.582 / 0.568 | 3 | **TODO** |

**How to read the scores.** With a single seed, validation differences below ~0.005 are within noise. `labeling_sample` contains only ~32 gold locations, so one location more or less moves its F1 by ~0.03. Differences of 0.05–0.10 there are only 2–3 locations.

---

## 8. Metrics and optimization curves

All curves are in the W&B project: <https://wandb.ai/lyaroshenko-kse/ua-locations-ner>

<!-- Export these panels from W&B (panel menu → "Download PNG") into images/ and keep the file names below. -->

| Curve | Figure |
|---|---|
| Train vs validation loss (baseline) | ![loss](images/loss.png) |
| Validation F1 per epoch, all experiments | ![val_f1](images/val_f1.png) |
| Total gradient norm (baseline) | ![grad_norm](images/grad_norm.png) |
| Learning rate, OneCycleLR (exp. d) | ![lr](images/lr.png) |

**Observations (baseline run):**

- **Convergence and overfitting.** Validation F1 peaks at **epoch 3 (0.910)**, then stops improving, and early stopping ends training after epoch 5. By then training loss is **0.003** and validation loss **0.019**, about 6× higher, a clear sign of overfitting. Checkpointing keeps the epoch 3 model.
- **Gradient flow is healthy.** Per-layer gradient norms decrease smoothly as the loss falls, with no spikes (no exploding gradients) and no layer collapsing to zero early (no vanishing gradients). The total norm stayed well below the clipping threshold of 1.0, so clipping acted only as a safety net.
- **Pretrained embeddings converge faster.** The fastText model reached its best validation F1 already at **epoch 1**, against epoch 3 for random initialization.
- **OneCycleLR** needed the full schedule (best epoch 4, no early stopping) to reach the same F1 as plain Adam.

---

## 9. Model insights

1. **Bidirectionality matters most.** Removing the backward direction costs **−0.020** validation F1, the only clearly harmful change. Whether a word is a location often depends on the words *after* it (e.g. "в Орловской **областях**", "на **вулиці** …" shows up on the left, but "… **районі**" on the right).
2. **Pretrained knowledge helps a little.** fastText initialization gave the best validation F1 (+0.005, at the edge of noise) and the fastest convergence, despite covering only Ukrainian words (Russian words stay random).
3. **The embedding table is the bottleneck, not the RNN.** 95% of parameters sit in word embeddings, and overfitting begins after ~3 epochs. BPE reaches almost the same F1 (0.907) with a **12× smaller** embedding table, and GRU (fewer parameters than LSTM) matches the LSTM. Extra RNN capacity brings little.
4. **Optimizer choice is secondary.** AdamW + OneCycleLR matched Adam (0.910 vs 0.910) at a higher training cost.
5. **Strict decoding ≥ loose decoding** for every model. The model's B/I distinction is reliable enough that merging adjacent entities does more harm than good.
6. **Validation does not predict test-like performance.** On `labeling_sample` all models drop to 0.55–0.66 F1, and the ranking changes: the 1-layer model is best there but second-worst on validation. The main error sources, judging by the sample, are:
   - streets, addresses and districts that the training data rarely label as locations;
   - long, multi-line posts unlike the training sentences;
   - rare place names that fall out of the word vocabulary as `<unk>`.

---

## 10. Conclusions

- A simple **word-level BiLSTM tagger** trained on ~1.2M Ukrainian and Russian sentences reaches **0.91 F1** on in-domain validation using only recurrent layers.
- The best configuration is a **2-layer BiLSTM with fastText-initialized embeddings** (val F1 0.915). We use it for the final submission. **TODO: confirm against the leaderboard.**
- Architecture details (LSTM vs GRU, 1 vs 2 layers, optimizer schedule) change validation F1 by at most ~0.01, while **bidirectionality** and **pretrained embeddings** are the choices that matter.
- The biggest remaining gap is not model capacity but the **difference between the training data and the test data**: clean news sentences versus Telegram posts, and city/region labels versus street/address labels. This shows as a drop from ~0.91 (validation) to ~0.6 (test-like sample).

### Limitations and possible next steps
- **Single seed per experiment;** repeating the top 2–3 configurations with 3 seeds would show which differences are real.
- **Labeling mismatch:** adding training examples labeled like the test set (streets and addresses), even a few hundred, would likely help more than any architecture change.
- **Unknown words:** combining word embeddings with a character-level BiLSTM per word would handle rare and misspelled place names while staying fully recurrent.
- **Structured decoding:** a CRF output layer (still no attention) would enforce valid BIO sequences.
- **Ensembling:** averaging the saved softmax probabilities of several models before decoding, and tuning a probability threshold on `labeling_sample`.

---

## 11. How to reproduce

**Requirements:** Python 3.10+, `torch`, `lightning`, `wandb`, `pandas`, `numpy`, `scikit-learn`, `tokenizers` (only for experiment f), `matplotlib`.

```bash
pip install torch lightning wandb pandas numpy scikit-learn tokenizers matplotlib
```

1. Put the competition files into `data/`.
2. Open the notebook in Colab (GPU runtime, High RAM on) and run the sections in order:
   1. **EDA:** statistics and plots.
   2. **Split:** builds the training pool and saves `data/pool.parquet` + `data/split.npz`.
   3. **Preprocessing:** tokenizer, BIO tags, vocabulary, `Dataset` / `collate_fn`.
   4. **Model + training:** `Tagger` LightningModule and the Trainer with early stopping and checkpointing. Log in to W&B when prompted.
   5. **Prediction:** line splitting, decoding and `submission.csv`.
   6. **Experiments:** runs the baseline and experiments (a)–(f), writing `results.csv` and one `submission_<name>.csv` per run.
3. Training uses seed 42 throughout. The vocabulary is saved to `vocab.pkl` (and `bpe.json` for experiment f), because prediction needs the exact same vocabulary.

## Repository structure

```
.
├── README.md
├── notebook.ipynb          # TODO: your notebook name
├── results.csv             # experiment table
└── images/                 # curves exported from W&B
```
