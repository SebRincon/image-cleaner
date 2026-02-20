# Workflow for Removing Images Containing People from an S3 Dataset Using CLIP on Runpod

## Objective, inputs, and outputs

You want an automated, reproducible workflow that scans all images under the `train/` prefix in your S3 bucket, flags any image that contains a person (or “people”), and creates two deliverables:

A cleaned dataset written back into the *same* bucket under a new prefix that you’re calling `/cleaned` (in S3 terms: a **key prefix**, not a real directory), and a removal manifest listing exactly which image keys (IDs) should be removed (optionally including usable URLs). In Amazon S3, the console’s “folders” are a visualization built from shared key-name prefixes and delimiters like `/`; the storage model itself is flat, and “folders” are prefixes. citeturn10search0turn10search1turn10search3

A practical “cleaned” layout that avoids collisions and keeps provenance is:

- Source objects: `train/<filename>.jpg`
- Cleaned objects: `cleaned/train/<filename>.jpg`
- Manifests (CSV/JSON): `cleaned/manifests/<run_id>/*.csv` and `cleaned/manifests/<run_id>/run_metadata.json`

This structure makes it easy to rerun with different thresholds/models without overwriting prior results, and it keeps the cleaned dataset parallel to the original `train/` prefix. citeturn10search0turn10search9

## Detection strategy: what CLIP can do well, and how to make it reliable for “person present”

### What CLIP is actually doing

CLIP (Contrastive Language–Image Pretraining) is trained on large-scale (image, text) pairs to align image and text embeddings in a shared space, enabling **zero-shot** classification by comparing an image embedding to embeddings of text prompts. citeturn0search2turn0search10turn8view0

In practice, you:
- Encode candidate text prompts (e.g., “a photo of a person”)
- Encode an image
- Compute similarity (often cosine similarity, scaled) and optionally apply a softmax over candidate prompts to get “probabilities” across the text set citeturn8view0turn7view0

This makes CLIP usable as a **binary filter** (“person” vs “no person”) even though it is not a bounding-box detector. The model is designed to rank/score text relevance to the image, not to localize objects. citeturn0search2turn8view0

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["CLIP contrastive learning image text diagram","OpenAI CLIP architecture figure","OWL-ViT open vocabulary object detection example"] ,"num_per_query":1}

### Making “person present” robust with prompt sets (not single prompts)

A single prompt like “a photo of a person” can be brittle. A more reliable approach for dataset filtering is to use **prompt ensembles**, where you average text embeddings across multiple phrasings for the positive concept and (optionally) multiple phrasings for a negative concept, then compare. This aligns with how CLIP is typically used for zero-shot prediction: image features and text features are normalized and compared to produce similarity scores. citeturn8view0turn7view0

A strong starting prompt set for “people present” (positive) is:
- “a photo of a person”
- “a photo of a human”
- “a photo of people”
- “a photo of a man”
- “a photo of a woman”
- “a photo of a child”
- “a portrait of a person”

And for “no people” (negative), prompts like:
- “a photo of a landscape”
- “a photo of a building”
- “a photo of food”
- “a photo of an object”
- “an empty room”

Then you compute a binary score such as:

- `people_prob = softmax([sim_pos, sim_neg])[0]`  
  (i.e., probability mass assigned to the positive prompt set when competing against the negative set)

This leverages CLIP’s intended “most relevant text snippet given an image” usage pattern. citeturn8view0turn1search0

### When you should add a second-stage check

If your requirement is “remove anything with a real human” (including small/partial faces, reflections, posters, statues, mannequins), CLIP-only filtering may produce false positives/negatives depending on the domain.

Two common hardening options:

- **Open-vocabulary detection** (localization) with OWL-ViT, queried with text like “person” / “human face.” OWL-ViT is explicitly an open-vocabulary *object detection* model that can detect objects described by text queries. citeturn2search0turn2search0  
- **Managed detection** with Amazon Rekognition:
  - `DetectFaces` returns face bounding boxes/confidence (up to the 100 largest faces). citeturn9search1turn9search18
  - `DetectLabels` can return object labels and includes instance bounding boxes for common objects such as people. citeturn9search14turn9search3  
  This option is operationally simple if your data is already in S3, but it’s a paid API (pricing is usage-based). citeturn9search2

Because you explicitly asked for CLIP, the workflow below is CLIP-first, with clear points where you can optionally add OWL-ViT or Rekognition as a confirmatory step.

## Secure access and execution on Runpod with AWS credentials

### Credential handling: do not embed keys in code or images

For AWS SDK for Python (boto3), the standard pattern is to supply credentials via environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_SESSION_TOKEN` for temporary creds). Boto3 explicitly documents that it checks these environment variables for credentials. citeturn1search2turn1search9

AWS security guidance strongly prefers **temporary credentials** (for example via roles) over long-lived access keys where feasible, and recommends key rotation when long-lived keys are unavoidable. citeturn3search1turn3search9turn1search13

Runpod’s own guidance is consistent with this: use secrets management / environment variables and avoid hardcoding secrets. Runpod documents “Never hardcode secrets: Use Runpod secrets for sensitive data.” citeturn3search2turn3search6

### Runpod execution model considerations

If you plan to use a custom container image, note that Runpod manages the Docker daemon in a way that prevents running Docker-in-Docker inside a pod; the recommended workaround is to create a custom template with the image you need. citeturn0search11

This matters for reproducibility: in practice, you’ll either:
- Use a Runpod base image and `pip install` runtime deps, or
- Build and publish a container image that already includes PyTorch + CLIP dependencies.

### Minimal S3 permissions for this job

At a minimum, the runner needs to:

- List keys under `train/` (bucket-level `ListBucket`, typically constrained by prefix conditions)
- Read objects under `train/` (`GetObject`)
- Write manifests under `cleaned/manifests/`
- Copy or upload kept images into `cleaned/train/`

S3 supports restricting listing to a prefix via policy conditions using `s3:prefix`. citeturn1search3turn10search6

## Step-by-step workflow

### Define your policy and choose a run ID

Decide what “contains people” means for your use case. For example:
- Strict: any face/body/partial human → remove
- Lenient: only prominent humans → remove

Pick a `run_id` like `2026-02-20_clip_peoplefilter_v1` so artifacts are traceable.

### Dry-run calibration on a sample

Before copying anything, run a dry pass that:
- Samples (e.g., 200–1,000) images from `train/`
- Computes CLIP “people” scores
- Writes out a ranked CSV and optionally exports a small set of borderline images for manual review

This aligns with the reality that CLIP is a zero-shot model controlled by prompts and score thresholds; calibration is how you translate similarity into a reliable binary filter. citeturn8view0turn2search2

### Full scan + copy kept images + write manifests

For the full run:

1. List all objects under `train/` using pagination (S3 list calls return up to 1,000 keys per page). citeturn0search4turn10search6  
2. For each image:
   - Download bytes
   - Decode and preprocess for CLIP
   - Compute “people probability” from your prompt sets
3. If “people present” → add key to `removed.csv`
4. If “no people” → copy object to `cleaned/train/...` and add key to `kept.csv`
5. Upload run metadata JSON documenting:
   - Model name + pretrained weights
   - Prompt sets
   - Threshold
   - Timestamp, code version, and counts

S3 listing and read-after-write/list consistency is now strong, simplifying verification that newly written objects appear immediately in list operations. citeturn3search0turn3search3

### Generate URLs (only if needed)

Your manifest can include:
- `s3_uri` like `s3://bucket/train/...`
- A virtual-hosted-style HTTPS URL in the documented form `https://bucket-name.s3.region-code.amazonaws.com/key-name` citeturn0search1turn0search13

However, if the bucket/objects are not public (and most buckets default to blocking public access), those HTTPS URLs won’t be fetchable anonymously; you’ll need **presigned URLs** for shareable links. AWS supports presigned URLs as time-limited access without changing bucket policies, and boto3 can generate them via `generate_presigned_url`. citeturn4search1turn4search0  
Also note S3 Block Public Access settings can override public policies/ACLs and are enabled by default for new buckets. citeturn4search3turn4search11

## Reference implementation blueprint

What follows is a practical implementation approach you can run on a GPU pod. It uses OpenCLIP because it provides a well-supported open-source CLIP implementation and demonstrates the normalized-embedding + similarity workflow. citeturn7view0turn8view0

### Dependencies

Use a PyTorch-enabled environment and install:

```bash
pip install -U boto3 pillow pandas tqdm open_clip_torch
```

OpenCLIP documents loading models via `open_clip.create_model_and_transforms` and computing normalized image/text features with similarity-based softmax. citeturn7view0

### Python script: scan `train/`, write `cleaned/`, produce manifests

```python
import argparse
import io
import json
import os
from datetime import datetime, timezone

import boto3
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def s3_https_url(bucket: str, region: str, key: str) -> str:
    # Virtual-hosted-style URL pattern (may require auth / object could be private).
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--region", required=True)

    p.add_argument("--src-prefix", default="train/")
    p.add_argument("--dst-prefix", default="cleaned/train/")
    p.add_argument("--manifest-prefix", default="cleaned/manifests/")

    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.50)

    p.add_argument("--max-images", type=int, default=0, help="0 = no limit")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--emit-presigned-urls", action="store_true")
    p.add_argument("--presign-exp-seconds", type=int, default=7 * 24 * 3600)

    return p.parse_args()


def list_s3_keys(s3_client, bucket: str, prefix: str):
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj.get("Size", None), obj.get("ETag", None), obj.get("LastModified", None)


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Boto3 will read credentials from env vars (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.).
    s3 = boto3.client("s3", region_name=args.region)

    # Load CLIP model (+ preprocessing transform)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()

    # Prompt sets
    pos_prompts = [
        "a photo of a person",
        "a photo of a human",
        "a photo of people",
        "a photo of a man",
        "a photo of a woman",
        "a photo of a child",
        "a portrait of a person",
    ]
    neg_prompts = [
        "a photo of a landscape",
        "a photo of a building",
        "a photo of food",
        "a photo of an object",
        "an empty room",
    ]

    # Encode prompts once
    with torch.no_grad():
        pos_tokens = open_clip.tokenize(pos_prompts).to(device)
        neg_tokens = open_clip.tokenize(neg_prompts).to(device)

        pos_text = model.encode_text(pos_tokens)
        neg_text = model.encode_text(neg_tokens)

        pos_text = pos_text / pos_text.norm(dim=-1, keepdim=True)
        neg_text = neg_text / neg_text.norm(dim=-1, keepdim=True)

        # Average to get one vector per class
        pos_vec = pos_text.mean(dim=0, keepdim=True)
        neg_vec = neg_text.mean(dim=0, keepdim=True)
        pos_vec = pos_vec / pos_vec.norm(dim=-1, keepdim=True)
        neg_vec = neg_vec / neg_vec.norm(dim=-1, keepdim=True)

        class_text = torch.cat([pos_vec, neg_vec], dim=0)  # [2, d]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_base = f"{args.manifest_prefix.rstrip('/')}/{run_id}"
    removed_rows = []
    kept_rows = []
    errors_rows = []

    # Collect candidate keys
    all_keys = []
    for key, size, etag, last_modified in list_s3_keys(s3, args.bucket, args.src_prefix):
        if key.lower().endswith(IMAGE_EXTS):
            all_keys.append((key, size, etag, last_modified))
        if args.max_images and len(all_keys) >= args.max_images:
            break

    def dst_key_from_src(src_key: str) -> str:
        if not src_key.startswith(args.src_prefix):
            # Fallback: just append the full key
            return args.dst_prefix.rstrip("/") + "/" + src_key
        rel = src_key[len(args.src_prefix):]
        return args.dst_prefix + rel

    # Batch processing
    for i in tqdm(range(0, len(all_keys), args.batch_size), desc="Scanning"):
        batch = all_keys[i:i + args.batch_size]

        imgs = []
        meta = []
        for key, size, etag, last_modified in batch:
            try:
                obj = s3.get_object(Bucket=args.bucket, Key=key)
                b = obj["Body"].read()
                im = Image.open(io.BytesIO(b)).convert("RGB")
                imgs.append(preprocess(im))
                meta.append((key, size, etag, last_modified))
            except Exception as e:
                errors_rows.append({
                    "key": key,
                    "error": repr(e),
                })

        if not imgs:
            continue

        image_tensor = torch.stack(imgs, dim=0).to(device)

        with torch.no_grad():
            # Autocast can improve throughput on modern GPUs
            autocast_ctx = torch.autocast(device_type="cuda") if device == "cuda" else nullcontext()
            with autocast_ctx:
                img_feat = model.encode_image(image_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # Similarity scores vs [pos, neg]; softmax -> "people probability"
            logits = 100.0 * img_feat @ class_text.T
            probs = logits.softmax(dim=-1)[:, 0].detach().cpu().numpy()

        for (key, size, etag, last_modified), people_prob in zip(meta, probs):
            row = {
                "key": key,
                "s3_uri": f"s3://{args.bucket}/{key}",
                "https_url": s3_https_url(args.bucket, args.region, key),
                "people_prob": float(people_prob),
                "size": size,
                "etag": etag,
                "last_modified": last_modified.isoformat() if last_modified else None,
            }

            if args.emit_presigned_urls:
                row["presigned_url"] = s3.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": args.bucket, "Key": key},
                    ExpiresIn=args.presign_exp_seconds,
                )

            if people_prob >= args.threshold:
                removed_rows.append(row)
            else:
                kept_rows.append(row)
                if not args.dry_run:
                    dst_key = dst_key_from_src(key)
                    # Server-side copy avoids uploading bytes back from the pod.
                    s3.copy_object(
                        Bucket=args.bucket,
                        Key=dst_key,
                        CopySource={"Bucket": args.bucket, "Key": key},
                    )

    # Write manifests locally
    removed_df = pd.DataFrame(removed_rows).sort_values("people_prob", ascending=False)
    kept_df = pd.DataFrame(kept_rows).sort_values("people_prob", ascending=False)
    errors_df = pd.DataFrame(errors_rows)

    removed_csv = removed_df.to_csv(index=False).encode("utf-8")
    kept_csv = kept_df.to_csv(index=False).encode("utf-8")
    errors_csv = errors_df.to_csv(index=False).encode("utf-8")

    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "region": args.region,
        "src_prefix": args.src_prefix,
        "dst_prefix": args.dst_prefix,
        "manifest_prefix": manifest_base,
        "model": {"name": args.model, "pretrained": args.pretrained},
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
        "counts": {
            "scanned_images": len(all_keys),
            "kept": len(kept_rows),
            "removed": len(removed_rows),
            "errors": len(errors_rows),
        },
        "prompts": {"positive": pos_prompts, "negative": neg_prompts},
    }
    metadata_json = json.dumps(metadata, indent=2).encode("utf-8")

    # Upload manifests to S3
    s3.put_object(Bucket=args.bucket, Key=f"{manifest_base}/removed.csv", Body=removed_csv)
    s3.put_object(Bucket=args.bucket, Key=f"{manifest_base}/kept.csv", Body=kept_csv)
    s3.put_object(Bucket=args.bucket, Key=f"{manifest_base}/errors.csv", Body=errors_csv)
    s3.put_object(Bucket=args.bucket, Key=f"{manifest_base}/run_metadata.json", Body=metadata_json)

    print(json.dumps(metadata, indent=2))


# For Python <3.7 compatibility you can remove nullcontext usage.
from contextlib import nullcontext

if __name__ == "__main__":
    main()
```

Key points this script implements:

- Uses `list_objects_v2` pagination via boto3 paginator (important because a single list call returns up to 1,000 objects). citeturn0search4turn0search16  
- Uses OpenCLIP’s standard normalized-embedding similarity approach and logits softmax (mirroring the common CLIP usage shown in both the original CLIP repo and OpenCLIP examples). citeturn8view0turn7view0  
- Writes manifests to a unique `run_id` path for traceability.
- Uses server-side `copy_object` for kept images (CopyObject is an S3 API that creates a new object from an existing one; appropriate here because images are far below the single-call size limits documented for CopyObject). citeturn2search3turn2search12  
- Optionally generates presigned URLs with boto3’s `generate_presigned_url` API. citeturn4search0turn4search1

## Validation, cost, and operational considerations

### Verification checks that are easy and meaningful

After a full run, do three quick validations:

- Count parity: `kept + removed + errors == scanned_images` (this is already written to `run_metadata.json`).
- Spot-check a sample of “high people_prob” removed images and a sample of “low people_prob” kept images.
- If you have a “near-threshold” zone (e.g., 0.45–0.55), spot-check those specifically; that’s where most false positives/negatives will live.

Also, because S3 now provides strong consistency for GET/PUT/LIST, you can list the `cleaned/` prefix immediately after writing and expect an accurate reflection of what’s in the bucket. citeturn3search0turn3search3

### Be explicit about what “image URL” means in a private bucket

Even if you record an HTTPS-form object URL, it may not be publicly retrievable unless you intentionally grant public read and/or adjust Block Public Access. AWS documentation emphasizes that making buckets public requires deliberate configuration, and Block Public Access can prevent public access even if policies/ACLs would otherwise allow it. citeturn4search3turn4search2

For workflows like dataset review, presigned URLs are usually the right “shareable URL” primitive: they grant time-limited access without changing bucket policies, and they use the permissions of the principal that generated them. citeturn4search1turn4search9

### Data transfer cost awareness with Runpod + S3

If your Runpod worker is running outside AWS, downloading the entire dataset from S3 is effectively **data transfer out** from AWS to the internet, which is generally a billed category; AWS also describes that inbound data transfer is generally not charged, while outbound data transfer to the internet is charged per service/region. citeturn5search4turn6view0

If data transfer cost becomes material, the main lever is to run the scanning compute inside AWS (same region as the bucket) rather than pulling the full dataset out to an external GPU provider.

### Security posture

Use Runpod secrets/env vars rather than baking credentials into images or code. citeturn3search2turn3search6  
Prefer temporary credentials / roles where possible and follow IAM best practices around credential management. citeturn3search1turn3search9turn1search13