## Model Preparation

For post-training, download the following pretrained models:

- Stable Diffusion v1.4: `CompVis/stable-diffusion-v1-4`
- CLIP ViT-L/14: `openai/clip-vit-large-patch14`

After downloading, update the corresponding model paths in:

- `aesthetic_scorer.py`
- `main.py`

## Environment Setup (Text-to-Image)

The experiments are tested with:

- `trl==0.12.2`
- `transformers==4.46.3`

```bash
cd ./code
conda create -n RLR python=3.10 -y
conda activate RLR

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu122
pip install -r requirements.txt
```

## Training Commands

Run PPO baseline (RL):

```bash
bash ./scripts/aesthetic_ppo.sh
```

Run Stable Diffusion v1.4 training with the RLR optimizer:

```bash
bash ./scripts/aesthetic_RLR_vb.sh
```

## Notes

To train with truncated backpropagation (AlignProp style), set `chain_len=0` in `scripts/aesthetic_RLR_vb.sh`.

