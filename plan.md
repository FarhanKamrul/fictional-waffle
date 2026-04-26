# Surgical Safety Steering: Attribution-Guided Neuron Masking in Diffusion Model Text Encoders

## Step 0: Environment & Workspace Setup (Remote Server)

> **Principle**: Everything that is expensive to download or compute is stored under `/workspace` and guarded by existence-checks so it is never repeated across sessions.

### 0.1 — Workspace Directory Structure

```
/workspace/
├── models/
│   ├── stable-diffusion-v1-4/           # SD 1.4 weights (HuggingFace, ~4 GB)
│   ├── clip-vit-large-patch14/          # CLIP text encoder (HuggingFace, ~890 MB)
│   ├── sae_checkpoint_layer{L*}/        # k-SAE — name set AFTER Step 1 selects L*
│   │   ├── final/                       # Loaded by SparseAutoencoder.load_from_disk()
│   │   └── train_config.json            # Saved hyperparams + L* for reproducibility
│   └── classifiers/
│       ├── nudenet/                     # NudeNet detector weights
│       └── q16/                         # Q16 unsafe-image classifier weights
│
├── datasets/
│   ├── i2p/
│   │   ├── i2p_4703_prompts.csv         # Full I2P benchmark (download once)
│   │   └── by_category/                 # Per-category splits (generated in Step 4 pre-processing)
│   │       ├── nudity.txt
│   │       ├── violence.txt
│   │       └── ...                      # one file per I2P category
│   ├── coco/
│   │   ├── annotations/                 # COCO 2017 val captions (download once)
│   │   └── val2017/                     # COCO 2017 val images, ~1 GB (download once)
│   ├── sae_train_corpus/
│   │   └── train_prompts.txt            # ~500k LAION/CC3M captions (download once)
│   └── custom_prompts/
│       └── extended_categories.csv      # Claude-generated prompts for underrepresented categories
│
├── activations/                         # Pre-computed & cached — never recompute if file exists
│   ├── layer_analysis/                  # Step 1: used for layer selection PCA
│   │   ├── concept_acts_layer6.pt
│   │   ├── concept_acts_layer9.pt
│   │   ├── concept_acts_layer11.pt
│   │   ├── concept_acts_layer12.pt
│   │   ├── neutral_acts_layer6.pt
│   │   ├── neutral_acts_layer9.pt
│   │   ├── neutral_acts_layer11.pt
│   │   └── neutral_acts_layer12.pt
│   ├── sae_train_acts_layer{L*}.pt      # Step 0.3: SAE training data at L*
│   └── by_category/                     # Step 4A: attribution scoring activations
│       ├── nudity_concept_acts.pt
│       ├── nudity_neutral_acts.pt
│       └── ...
│
├── masks/                               # Two designs per category (Step 4C)
│   ├── nudity_mask_soft.pt
│   ├── nudity_mask_binary.pt
│   ├── violence_mask_soft.pt
│   └── ...
│
├── fid_features/
│   └── coco_val_inception_features.npz  # Pre-computed once before Phase B (Step 5)
│
├── generated_images/                    # All 5k images, one folder per condition
│   ├── original/
│   ├── vanilla/
│   ├── soft_attenuation/
│   └── binary_compensated/
│
├── metrics/
│   └── results.json                     # Aggregated ASR / FID / CLIP Score / LPIPS / PAPS
│
└── code/
    └── steerers/                        # Git clone of the Concept Steerers repo
```

### 0.2 — One-Time Asset Downloads (idempotent shell script)

Run `setup_workspace.sh` on first SSH login. Every block checks existence before acting.

```bash
#!/bin/bash
# setup_workspace.sh — run once on the remote server, safe to re-run.

WORKSPACE=/workspace
mkdir -p $WORKSPACE/{models/classifiers/{nudenet,q16},datasets/{i2p/by_category,coco,sae_train_corpus,custom_prompts},activations/{layer_analysis,by_category},masks,fid_features,generated_images/{original,vanilla,soft_attenuation,binary_compensated},metrics,code}

# ── 1. Steerers repo ────────────────────────────────────────────────────────
if [ ! -d "$WORKSPACE/code/steerers" ]; then
  git clone https://github.com/kim-dahye/steerers.git $WORKSPACE/code/steerers
fi

# ── 2. HuggingFace model weights (cached to /workspace/models) ──────────────
python - <<'EOF'
import os
from huggingface_hub import snapshot_download

MODELS = [
    ("CompVis/stable-diffusion-v1-4", "stable-diffusion-v1-4"),
    ("openai/clip-vit-large-patch14", "clip-vit-large-patch14"),
]
for repo_id, folder in MODELS:
    dest = f"/workspace/models/{folder}"
    if not os.path.isdir(dest):
        print(f"Downloading {repo_id}...")
        snapshot_download(repo_id=repo_id, local_dir=dest)
    else:
        print(f"✓ {folder} already present, skipping.")
EOF

# ── 3. I2P benchmark prompts ────────────────────────────────────────────────
if [ ! -f "$WORKSPACE/datasets/i2p/i2p_4703_prompts.csv" ]; then
  # Download from the official I2P HuggingFace dataset
  python -c "
from datasets import load_dataset
ds = load_dataset('AIML-TUDA/i2p-benchmark', split='train')
ds.to_csv('/workspace/datasets/i2p/i2p_4703_prompts.csv', index=False)
print('✓ I2P dataset saved.')
"
else
  echo "✓ I2P prompts already present, skipping."
fi

# ── 4. COCO 2017 val images & annotations ───────────────────────────────────
COCO_ANN=$WORKSPACE/datasets/coco/annotations/captions_val2017.json
if [ ! -f "$COCO_ANN" ]; then
  wget -q -P /tmp http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q /tmp/annotations_trainval2017.zip -d $WORKSPACE/datasets/coco/
  rm /tmp/annotations_trainval2017.zip
  echo "✓ COCO annotations saved."
else
  echo "✓ COCO annotations already present, skipping."
fi

COCO_IMGS=$WORKSPACE/datasets/coco/val2017
if [ ! -d "$COCO_IMGS" ] || [ -z "$(ls -A $COCO_IMGS 2>/dev/null)" ]; then
  wget -q -P /tmp http://images.cocodataset.org/zips/val2017.zip
  unzip -q /tmp/val2017.zip -d $WORKSPACE/datasets/coco/
  rm /tmp/val2017.zip
  echo "✓ COCO val images saved."
else
  echo "✓ COCO val images already present, skipping."
fi

echo ""
echo "=== Workspace setup complete ==="
```

### 0.3 — SAE Training ⚠️ *Deferred: run AFTER Step 1 determines L\**

The k-SAE has **not yet been trained**. It must be trained on the layer `L*` identified in Step 1 — do not run this until the layer analysis is complete. Training is a three-stage pipeline:

#### Stage A — Prepare the training corpus

The SAE must learn a general dictionary of text-encoder concepts, so we train it on a large, neutral set of captions (not the I2P prompts) — e.g., a subset of LAION-Aesthetics or CC3M captions.

```bash
# Check if corpus already exists before downloading
if [ ! -f "$WORKSPACE/datasets/sae_train_corpus/train_prompts.txt" ]; then
  python $WORKSPACE/code/steerers/scripts/prepare_sae_corpus.py \
    --output $WORKSPACE/datasets/sae_train_corpus/train_prompts.txt \
    --num_prompts 500000
  echo "✓ SAE training corpus ready."
else
  echo "✓ SAE training corpus already present, skipping."
fi
```

#### Stage B — Extract CLIP Layer L\* activations for SAE training

Run all ~500k training prompts through the CLIP text encoder and save the hidden states at the winning layer `L*`. This is the actual data the SAE trains on.

```bash
# Set L_STAR to the winning layer index from Step 1 before running this.
L_STAR=<result_of_step_1>   # e.g. 9, 10, 11 — determined empirically

if [ ! -f "$WORKSPACE/activations/sae_train_acts_layer${L_STAR}.pt" ]; then
  python $WORKSPACE/code/steerers/scripts/extract_train_activations.py \
    --prompts   $WORKSPACE/datasets/sae_train_corpus/train_prompts.txt \
    --clip_path $WORKSPACE/models/clip-vit-large-patch14 \
    --layer     $L_STAR \
    --out       $WORKSPACE/activations/sae_train_acts_layer${L_STAR}.pt \
    --batch_size 256
  echo "✓ SAE training activations extracted (Layer ${L_STAR})."
else
  echo "✓ SAE training activations already present, skipping."
fi
```

#### Stage C — Train the k-SAE

```bash
if [ ! -d "$WORKSPACE/models/sae_checkpoint_layer${L_STAR}/final" ]; then
  python $WORKSPACE/code/steerers/scripts/train_sae.py \
    --acts_path  $WORKSPACE/activations/sae_train_acts_layer${L_STAR}.pt \
    --input_dim  768    \  # CLIP ViT-L/14 hidden size (fixed regardless of layer)
    --hidden_dim 3072   \  # 4× expansion factor
    --k          32     \  # top-k sparsity (from Concept Steerers paper)
    --lr         4e-4   \  # confirmed working in prototype
    --steps      10000  \  # 10k steps as per Concept Steerers
    --bs         4096   \  # batch size confirmed working
    --out_dir    $WORKSPACE/models/sae_checkpoint_layer${L_STAR}
  echo "✓ SAE training complete. Checkpoint saved at layer ${L_STAR}."
else
  echo "✓ SAE checkpoint already exists, skipping training."
fi
```

After training, verify and save config:
```bash
ls $WORKSPACE/models/sae_checkpoint_layer${L_STAR}/final/
python -c "
import json
cfg = {'L_STAR': $L_STAR, 'k': 32, 'hidden_dim': 3072, 'lr': 4e-4, 'steps': 10000, 'bs': 4096}
with open('$WORKSPACE/models/sae_checkpoint_layer${L_STAR}/train_config.json','w') as f: json.dump(cfg,f,indent=2)
print('Config saved.')
"
```

### 0.4 — Python Environment

```bash
# On the remote server (one-time):
conda create -n steerers python=3.10 -y
conda activate steerers
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Pin versions confirmed working in the Colab prototype:
pip install diffusers==0.31.0 transformers==4.46.1 tokenizers==0.20.1 huggingface-hub==0.26.2
pip install accelerate open_clip_torch SDLens
pip install pytorch-fid lpips nudenet scikit-image scikit-learn
pip install datasets matplotlib pandas tqdm
```

### 0.6 — Hardware: A100 40GB (1×, ~$0.72/h)

**Target GPU:** A100 40GB via cloud rental.

**Why A100 40GB over alternatives:**
*   SD 1.4 (FP16) + SAE + batch size 16 peaks at ~12GB VRAM — 40GB gives ample headroom.
*   The 80GB variant costs $1.29/h for headroom you will not use.
*   The H100 ($2.29/h) is ~2× faster but this project is not throughput-bottlenecked — it is iteration-bottlenecked.
*   The L40S (48GB, $0.91/h) is close but A100 has better tensor core utilisation for mixed training + inference workloads.

**Expected cost for the entire project:**

| Phase | Images | Batch | Time | Cost |
| :--- | :--- | :--- | :--- | :--- |
| SAE Training | — | 4096 activations | ~45 min | ~$0.55 |
| Prototype iteration (×2–3 runs) | 2,000 | 16 | ~12 min each | ~$0.30 total |
| Full 5k eval | 20,000 | 16 | ~2 h | ~$1.44 |
| **Total** | | | **~4 h** | **~$2.30** |

### 0.5 — Environment Variables (add to `~/.bashrc`)

```bash
export WORKSPACE=/workspace
export HF_HOME=/workspace/models          # HuggingFace caches to /workspace, not ~/.cache
export TRANSFORMERS_CACHE=/workspace/models
# NOTE: Set L_STAR after Step 1, then re-source ~/.bashrc before running Step 0.3
export L_STAR=<fill_after_step_1>
export SAE_CKPT=/workspace/models/sae_checkpoint_layer${L_STAR}
export I2P_CSV=/workspace/datasets/i2p/i2p_4703_prompts.csv
export COCO_DIR=/workspace/datasets/coco
export ACT_DIR=/workspace/activations
export MASK_DIR=/workspace/masks
export IMG_DIR=/workspace/generated_images
```

Setting `HF_HOME` and `TRANSFORMERS_CACHE` ensures that even if a script calls `from_pretrained("CompVis/stable-diffusion-v1-4")`, it resolves to the local copy on `/workspace` rather than hitting the internet.

---

## Phase 1


### Step 1: Determine optimal layer for concept-neutral separation

#### 1A — Extract activations at candidate layers

Before any analysis, extract CLIP text encoder activations at each candidate layer for both prompt sets. This is a non-trivial compute step (~200 I2P + 200 COCO prompts × 4 layers).

```python
for layer in [6, 9, 11, 12]:
    concept_acts = extract_activations(pipe, i2p_prompts[:200], layer)  # (200, 77, 768)
    neutral_acts = extract_activations(pipe, coco_prompts[:200], layer) # (200, 77, 768)
    torch.save(concept_acts, f"$ACT_DIR/layer_analysis/concept_acts_layer{layer}.pt")
    torch.save(neutral_acts, f"$ACT_DIR/layer_analysis/neutral_acts_layer{layer}.pt")
```

#### 1B — PCA and separation analysis

*   For each layer, run PCA on the concatenated concept + neutral activations.
*   Measure concept-vs-neutral separation (linear probe accuracy or silhouette score).
*   Select the layer with the highest separation as `L*`.
*   **Gate**: SAE training (Step 0.3) does not begin until `L*` is confirmed. Set `L_STAR` in `~/.bashrc` and re-source.
*   The winning layer and its separation plot becomes **Figure 1 panel A** of the paper.

### Step 2: Evaluation Strategy
*   **Target**: Full 5k evaluation with FID, ASR, LPIPS, and CLIP Score.
*   **Prototype runs first**: Before committing the full evaluation compute, run a 500-image prototype to catch bugs, validate mask logic, and tune `k` and `λ` parameters.
*   Prototype is a validation gate, not a substitute for the full eval.

### Step 3: Fix the Dataset
*   **Existing**: I2P benchmark (4,703 prompts, 7 unsafe categories: nudity, violence, disturbing, hate, political, self-harm, shocking). This is your primary source — widely cited, reviewer-accepted.
*   **Novel category extension**: Use Claude to generate ~200 additional prompts for 2–3 categories underrepresented in I2P (e.g., gore/body horror, dangerous activities).

## Phase 2

### Step 4: Extend attribution to multiple categories

#### 4A — Attribution Scoring (uses I2P + COCO)
*   Collect concept-positive prompts (I2P category subset) → run through CLIP Layer `L*` → **pass through SAE encoder** → record SAE activations → average across 77 tokens → shape `(N, 3072)`.
*   Collect neutral prompts (1,000 COCO captions) → same process to get shape `(N, 3072)`.
*   Compute per-neuron attribution score: `score_j = concept_mean_j - neutral_mean_j`
*   Save: `$ACT_DIR/{category}_neuron_scores.pt` — one score vector per category.

#### 4B — Steering Vector (derived from I2P category subset, not arbitrary labels)

*   The steering direction is computed from the **same I2P category subset prompts** used for attribution scoring — not from an arbitrarily defined short label.
*   The steering vector is computed directly in the SAE's feature space, but it's applied inside the generation loop via the SAE decoder:

```python
# During the forward pass hook on Layer L*:
# 1. Encode the current hidden state to get SAE features
z_current = sae.encoder(hidden - sae.pre_bias)
# 2. Subtract the steering vector in feature space, scaled by λ and the mask
z_steered = z_current + (mask * λ * steering_vec_sae_features)
# 3. Decode back to the dense embedding space
hidden_steered = sae.decoder(z_steered) + sae.pre_bias
```

*   This is consistent with the attribution scoring methodology and generalises cleanly across all 7+ I2P categories without requiring manual label authoring.

#### 4C — Build Two Mask Designs per Category

Both masks use the same `neuron_scores` from 4A. Build and save both:

**Mask 1 — Soft Attenuation** (current prototype approach):
```python
mask_soft = floor + (1.0 - floor) * normalize(neuron_scores)  # floor=0.3
# Values in [0.3, 1.0]. Shared neurons are steered at 30%, concept neurons at 100%.
```

**Mask 2 — Hard Binary + Compensated λ** (new approach):
```python
# 'k' is determined empirically in Phase A (e.g., from the score distribution elbow)
topk_indices = neuron_scores.topk(k=k).indices  # top-k concept-exclusive neurons
mask_binary = torch.zeros(3072)
mask_binary[topk_indices] = 1.0
# λ increased to -1.0 to compensate for protecting shared neurons.
# Shared neurons: completely untouched. Concept neurons: fully steered at higher strength.
```

*   Save: `$MASK_DIR/{category}_mask_soft.pt` and `$MASK_DIR/{category}_mask_binary.pt`

### Step 5: Two-Phase Evaluation

Four conditions run in a single loop per seed (Original is generated once and shared):

| Condition | Mask | λ | What it tests |
| :--- | :--- | :--- | :--- |
| **Original** | None | — | Baseline (unsafe) |
| **Vanilla** | All neurons = 1.0 | -0.5 | Concept Steerers baseline |
| **Soft Attenuation** | `[0.3, 1.0]` continuous | -0.5 | Current prototype |
| **Binary Compensated** | TopK=60 binary | -1.0 | Full composition preservation |

```python
for prompt in prompts:
    for seed in seeds:
        img_original = generate(prompt, seed, hook=None)                        # generated once
        img_vanilla  = generate(prompt, seed, hook=uniform_hook(λ=-0.5))
        img_soft     = generate(prompt, seed, hook=attenuation_hook(floor=0.3, λ=-0.5))
        img_binary   = generate(prompt, seed, hook=binary_hook(k=k, λ=-1.0))    # k tuned from Phase A
```

#### Phase A — Prototype (500 images, ~12 min, ~$0.14)
*   Run on nudity category only. Goal: validate both mask designs, tune `k` and `λ` for the binary mask.
*   Report ASR and LPIPS to pick the better mask design.
*   Fix the `k` and `λ` values before moving to Phase B.

#### Pre-computation for Metrics

Before running Phase B, compute the Inception features for the COCO 2017 val set once. This prevents redundant processing during FID calculation.
```bash
python -m pytorch_fid $WORKSPACE/datasets/coco/val2017 $WORKSPACE/fid_features/coco_val_inception_features.npz --save-stats
```

#### Phase B — Full Evaluation (5,000 images, ~2h, ~$1.44)
*   Run only after Phase A confirms the mask design. One run, no re-runs.
*   **Seed strategy**: `seed_start=42, seed_step=7` to ensure diversity.
*   **Safety & Preservation (I2P Prompts)**:
    *   Split 5,000 runs proportionally across all I2P categories.
    *   **ASR**: NudeNet/Q16 classifier → % unsafe per condition.
    *   **LPIPS**: Against `img_original` (same seed, same prompt) — primary preservation metric.
    *   **PAPS**: Radar chart evaluation for 5 attributes using CLIP/DINO similarity against `img_original`.
    *   **CLIP Score**: Prompt-image alignment using `open_clip`.
*   **General Generation Quality (COCO Prompts)**:
    *   Generate 5,000 images using neutral COCO prompts (not I2P prompts) to measure if the steering degrades the model globally.
    *   **FID**: Compare these COCO-prompted generations vs. pre-computed COCO real stats using `pytorch-fid`. (Comparing I2P generations to COCO real images is a domain mismatch and statistically invalid).

### Step 6: Baseline Comparison

**Strategy**: Use published numbers from the original papers where available. Only re-run baselines if their code is publicly available AND the published numbers use different evaluation settings than ours.

| Baseline | Source for numbers | Re-run needed? |
| :--- | :--- | :--- |
| **ESD** | Table 1, Gandikota et al. 2023 | No — use published ASR/FID on I2P |
| **SLD-Med / SLD-Max** | Table 2, Schramowski et al. 2023 | No — use published ASR |
| **SAeUron** | Tables from Giwon et al. ICML 2025 | Possibly — compare eval protocol |
| **Vanilla Concept Steerers** | Run ourselves | Yes — this is our direct ablation (mask = all 1s) |

*   **Vanilla Steerers** is already computed as a condition in Step 5 (mask = all neurons, λ=-0.5). No extra work needed.
*   For ESD/SLD/SAeUron, verify that published numbers were computed on I2P with comparable classifiers (NudeNet/Q16). If the evaluation protocol differs, note the discrepancy in the paper rather than re-running.

### Step 7: Build paper figures
*   **Neuron score distribution plot**: Histogram of all 3,072 attribution scores with ±2σ thresholds marked (you already have this from your Colab).
*   **Top-10 concept neurons visualised**: For each top neuron, show the max-activating text prompts from your concept-positive set (what text activates this neuron most?).
*   **Mask sparsity plot**: Show that only ~40–60 neurons (out of 3,072) are strongly concept-specific, making your intervention surgical by design.

### Step 8: Methodology Diagram
*   Create a visually appealing architecture diagram (e.g., in Figma/Draw.io) showing: Text Prompt → CLIP Layer L* → SAE Encoder → Masking & Steering → SAE Decoder → continuing diffusion process.
*   Highlight the binary mask targeting specific concept-exclusive neurons while protecting shared composition neurons.

### Step 9: Main Results Table
*   Format a LaTeX table presenting ASR, LPIPS, PAPS (averaged across the 5 attributes), CLIP Score, and FID.
*   Include the Original, Vanilla, and chosen selective mask conditions.

### Step 10: Main Paper

## Table X: Baselines

| Baseline | What it does | Reference |
| :--- | :--- | :--- |
| **ESD (Erased Stable Diffusion)** | Fine-tunes weights to erase concept | Gandikota et al. 2023 |
| **SLD-Med / SLD-Max** | Guidance-based safe latent diffusion | Schramowski et al. 2023 |
| **SAeUron** | SAE-based concept unlearning (closest to yours) | Giwon et al. ICML 2025 |
| **Vanilla Concept Steerers** | Full k-SAE steering without masking | Kim & Ghadiyaram 2025 |
| **Yours (Neuron-Selective)** | Masked k-SAE steering | This work |
