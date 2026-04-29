# Surgical Safety Steering (S3): Attribution-Guided Inpainting in SD 2.1

## Step 0: Environment & Workspace Setup (Remote Server)

> **Principle**: Everything that is expensive to download or compute is stored under `/workspace` and guarded by existence-checks so it is never repeated across sessions.

### 0.1 — Workspace Directory Structure

```
/workspace/
├── models/
│   ├── stable-diffusion-2-1/            # SD 2.1 weights (HuggingFace)
│   ├── stable-diffusion-2-inpainting/   # SD 2.1 Inpainting weights
│   ├── segformer-b2-clothes/            # SegFormer segmentation backbone
│   ├── clip-vit-large-patch14/          # CLIP weights for CMMD evaluation
│   ├── sae_checkpoint_layer{L*}/        # k-SAE — 1024 to 4096 expansion
│   └── classifiers/
│       ├── nudenet/                     # NudeNet detector weights
│       └── q16/                         # Q16 unsafe-image classifier weights
│
├── datasets/
│   ├── i2p/
│   │   └── i2p_4703_prompts.csv         # Full I2P benchmark
│   ├── coco/
│   │   ├── annotations/                 # COCO 2017 val captions
│   │   └── val2017/                     # COCO 2017 val images (5K subset for CMMD)
│   ├── ring-a-bell/                     # Adversarial prompts for H3 testing
│   └── sae_train_corpus/
│       └── train_prompts.txt            # ~500k LAION/CC3M captions
│
├── activations/                         # Pre-computed OpenCLIP-ViT-H/14 activations
│   ├── layer_analysis/                  # Layers 16-23 candidates
│   ├── sae_train_acts_layer{L*}.pt      
│   ├── attribution_scores/              # Mean causal output scores
│   └── bayesian_traces/                 # pymc MCMC traces for H4 probabilistic scoring
│
├── generated_images/                    # 768x768 outputs
│   ├── original/
│   ├── baseline_gloce/
│   ├── baseline_esd/
│   ├── masks/                           # KSAE heatmaps and SegFormer deterministic masks
│   └── s3_alpha_composited/
│
├── evaluation/
│   ├── unet_pca_maps/                   # Phase 6 Joint PCA Proof maps (H2)
│   ├── pareto_fronts/                   # pymoo NSGA-II optimization plots for lambda
│   ├── entanglement_graphs/             # networkx co-activation graphs for failure analysis
│   └── metrics_results.json             # ASR / SSIM_safe / CMMD 
└── code/
    └── steerers/                        # Concept Steerers baseline codebase
```

### 0.2 — Python Environment & Hardware

**Target GPU:** A100 40GB or 80GB via cloud rental. 768x768 generation with inpainting and segmentation models simultaneously in VRAM requires significant headroom.

```bash
conda create -n s3_env python=3.10 -y
conda activate s3_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.31.0 transformers==4.46.1 accelerate open_clip_torch
pip install nudenet scikit-image scikit-learn matplotlib pandas tqdm
# Newly imported math/optimization skills
pip install pymoo pymc networkx
```

---

## Phase 0: Literature & Setup (Annotated Bibliography)

Utilize the agent skills `literature-review` and `paper-lookup` to synthesize the three pillars of the research background. This will form the foundation of the paper's introduction.
1.  **Safety Interventions:** Review GLoCE, ESD-u, and Concept Steerers.
2.  **Mechanistic Interpretability:** Review SAE literature and causal attribution.
3.  **Spatial Control:** Review SegFormer, Alpha Compositing, and Inpainting logic.

---

## Phase 1: Feature Discovery & Layer Calibration ($L^*$)

SD 2.1 uses OpenCLIP-ViT-H/14, which has 24 layers and a hidden dimension of 1024. We must identify the optimal layer for semantic steering.

1.  **Extract Activations:** Extract layer 16 through 23 hidden states for I2P (Unsafe) and COCO (Neutral) prompts.
2.  **PCA & Probe:** Run Joint PCA on the concatenated activations to measure linear separability.
3.  **Selection:** Select $L^*$ where unsafe concept separation is highest before structural collapse begins.

---

## Phase 2: High-Dimensional Dictionary Learning (KSAE)

Train the Sparse Autoencoder on the 1024-dimensional space of $L^*$.

1.  **Extract:** Pass ~500k neutral prompts through OpenCLIP and cache $L^*$ activations.
2.  **Train:** 
    *   **Input Dim:** 1024
    *   **Hidden Dim:** 4096 (4x expansion)
    *   **Sparsity ($k$):** 32
3.  **Checkpoint:** Save the sparse dictionary for the attribution phase.

---

## Phase 3: Attribution (Bayesian Output-Score Routing)

*Crucial deviation from prior work: We measure causal influence using Bayesian confidence intervals, not just input activation.*

1.  **Input Scoring:** Record mean activations for KSAE features on I2P prompts.
2.  **Causal Output Scoring (`pymc`):** Use `pymc` to calculate Bayesian confidence intervals for the causal influence of top features across multiple generation noise seeds.
3.  **Selectivity Logic:** Identify **Concept-Exclusive** neurons (high reliable output score for unsafe concept, 0 for neutral) vs. **Shared Compositional** neurons.

---

## Phase 4: The S3 Pipeline (Oracle Localization & Pareto Optimization)

This is the core generative loop replacing global steering.

### 4A — Generation & Oracle Localization (The "WHERE")
1.  **Generate:** Create the base image `img_orig` at 768x768 using the unsafe prompt.
2.  **Oracle Mapping:** Map the Concept-Exclusive KSAE features to spatial attention maps (via PatchSAE logic) to find the rough region.
3.  **Optimization (`pymoo`):** Use `pymoo` to mathematically find the Pareto optimal threshold ($\lambda$) for the SegFormer mask generation, balancing ASR (Safety) and SSIM_safe (Fidelity).
4.  **SegFormer Validation:** Run `segformer-b2-clothes` with the optimal $\lambda$ to generate a pixel-perfect binary mask. Intersection of KSAE map and SegFormer map = `mask_final`.
5.  **Entanglement Mapping (`networkx`):** If `mask_final` covers >80% of the image, use `networkx` to map feature co-activations and diagnose inescapable pose collapse.

### 4B — Inpainting & Alpha Compositing (The "WHAT")
1.  **Inpaint:** Pass `img_orig` and `mask_final` to `stable-diffusion-2-inpainting` conditioned on a safe prompt (e.g., "fully clothed"). Result = `img_safe`.
2.  **Alpha Blend:**
    ```python
    img_final = (img_orig * (1 - mask_final)) + (img_safe * mask_final)
    ```
    *This guarantees the background and non-targeted pixels are 100% mathematically identical to `img_orig`.*

---

## Phase 5: Evaluation & Data Analysis

Evaluate 5,000 I2P images across the baseline conditions (Vanilla, ESD-u, Concept Steerers, GLoCE, S3). Use the `exploratory-data-analysis` skill to process `metrics_results.json` for statistical significance.

### Metrics Collection:
1.  **Safety:** ASR (Attack Success Rate via NudeNet/Q16) + CLIP Score drop.
2.  **Composition:** **SSIM_safe** — Structural Similarity Index computed *only* on pixels where `mask_final == 0`. Target is $\ge 0.95$.
3.  **Fidelity:** **CMMD** (CLIP Maximum Mean Discrepancy) on a 5,000 subset of COCO to prove general generation quality isn't globally degraded.
4.  **Robustness:** Test against Ring-A-Bell adversarial prompts.
5.  **Failure Mode:** Calculate percentage of prompts suffering from *Conceptual Entanglement*.

---

## Phase 6: Mechanistic Proof (The PnP Visualization)

Prove scientifically that S3 preserves structural composition using the `matplotlib` skill to generate high-quality visualization grids.

1.  **Hook the U-Net:** Attach forward hooks to the SD 2.1 decoder blocks (`up_blocks.0` through `3`) during generation of `img_orig` and `img_final`.
2.  **Extract PCA Maps:** Run Joint PCA on the feature maps at varying resolutions (12x12, 24x24, 48x48, 96x96).
3.  **Visual Proof Generation:** Generate a comparison grid demonstrating:
    *   **Layer 1 (Structural, 12x12):** Identical RGB maps between `img_orig` and `img_final` (0% layout shift).
    *   **Layer 11 (Semantic, 96x96):** RGB map shift strictly localized to the `mask_final` area. 

---

## Phase 7: Paper Drafting & Adversarial Peer-Review

Transition from code to publication using the final suite of agent skills.
1.  **Methodology Diagrams:** Use `markdown-mermaid-writing` to generate the S3 pipeline architecture diagram.
2.  **Adversarial Review:** Use the `peer-review` and `hypothesis-generation` skills to simulate a CVPR/ICLR reviewer, probing the draft for logical holes in the evaluation metrics.
3.  **Formatting:** Compile the final draft into standard conference format using `latex-posters`.
