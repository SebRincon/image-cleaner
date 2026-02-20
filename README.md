# S3 people-cleaning workflow (CLIP + RunPod)

This project provides a single script to:

1. Scan `s3://<bucket>/train/` images.
2. Score each image with CLIP for `"person"`-style concepts.
3. Write non-person images to `s3://<bucket>/cleaned/train/`.
4. Produce removal manifests that include image IDs and URLs for cleanup/review.

## 1) Set up environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

You need a RunPod pod with Python and enough GPU/CPU for CLIP inference.

## 2) Configure AWS credentials

Export credentials in your RunPod pod environment (or use RunPod Secrets):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...   # optional
export AWS_DEFAULT_REGION=us-east-1
```

## 3) Run the filter

```bash
python scripts/clean_people_s3.py \
  --bucket hackathon-lens-correction-submissions \
  --region us-east-1 \
  --src-prefix train/ \
  --dst-prefix cleaned/train/ \
  --manifest-prefix cleaned/manifests/ \
  --threshold 0.50 \
  --batch-size 16 \
  --max-images 0
```
### Run in parallel across multiple pods

Use the same command with shard flags. Example for **4 pods**:

```bash
for i in 0 1 2 3; do
  nohup python scripts/clean_people_s3.py \
    --bucket hackathon-lens-correction-submissions \
    --region us-east-1 \
    --src-prefix train/ \
    --dst-prefix cleaned/train/ \
    --manifest-prefix cleaned/manifests/ \
    --run-id 2026-02-20_peopleclean_v1 \
    --shard-index $i \
    --shard-count 4 \
    --batch-size 16 \
    --threshold 0.50 \
    > /tmp/clean_shard_${i}.log 2>&1 &
done
```

Each pod writes outputs to a unique manifest folder:

`cleaned/manifests/2026-02-20_peopleclean_v1/shard-0000-of-0004/...`

`cleaned/manifests/2026-02-20_peopleclean_v1/shard-0001-of-0004/...`

... so they do not overwrite each other.

Useful flags:

- `--dry-run` to generate classification results without writing to `cleaned/`.
- `--run-id 2026-02-20_clip_v1` to choose a fixed run folder.
- `--emit-presigned-urls` if you want temporary review links in manifest rows.
- `--max-images 200` for calibration on a sample.

## 4) Outputs to expect

Run outputs are written under:

`s3://hackathon-lens-correction-submissions/cleaned/manifests/<run-id>/`

Files created:

- `removed.csv` → `image_id`, `s3_key`, `s3_uri`, `https_url`, `presigned_url` (optional), `people_prob`
- `kept.csv` → same schema for retained images
- `removed_image_ids.txt` → list of image IDs flagged for removal
- `removed_s3_urls.txt` → list of `s3://...` URLs for removal
- `errors.csv` → processing/copy errors
- `run_metadata.json` → threshold, model, count summary

The cleaned set is written here:

`s3://hackathon-lens-correction-submissions/cleaned/train/`

## Merge shard manifests after all pods finish

After 8 shard pods complete, run:

```bash
export RUN_ID=2026-02-20_peopleclean_v1
export SHARD_COUNT=8
bash merge_shard_manifests.sh
```

This creates:

- `s3://hackathon-lens-correction-submissions/cleaned/manifests/<run-id>/aggregate/removed_all.csv`
- `s3://hackathon-lens-correction-submissions/cleaned/manifests/<run-id>/aggregate/removed_all_image_ids.txt`
- `s3://hackathon-lens-correction-submissions/cleaned/manifests/<run-id>/aggregate/removed_all_s3_urls.txt`
- `s3://hackathon-lens-correction-submissions/cleaned/manifests/<run-id>/aggregate/merge_metadata.json`

## Choosing pods for ~20k images under 1 hour

Start with these practical profiles on `us-east-1` pods:

- Fast baseline: 4x A100/L4-class pod instances (24-40 GB VRAM) if budget allows.
- Cost efficient: 4x A10/A4000-class.
- Avoid running all shards on spot unless you can resume failed shards.

If 20k images are similar size and preprocessing overhead is normal for CLIP ViT-B-32, a 4-way shard with 24 GB GPUs is a reasonable target for <1 hour with network bandwidth as the main variable.

## 5) Example cleanup step

After review, remove flagged originals (if you choose) using your existing AWS process:

- Either delete the keys in `removed.csv` from `train/` manually.
- Or keep originals and treat `removed.csv` as an exclusion list in downstream training code.
