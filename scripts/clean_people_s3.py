#!/usr/bin/env python3
"""Filter an S3 training prefix with CLIP and write cleaned data + removal manifests."""

import argparse
import csv
import io
import json
import math
import os
from contextlib import nullcontext
from datetime import datetime, timezone
import zlib
from typing import Dict, Iterable, List, Tuple

import boto3
from botocore.config import Config
from PIL import Image
from tqdm import tqdm

import open_clip
import torch


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove images containing people from an S3 prefix using CLIP."
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name.")
    parser.add_argument("--region", required=True, help="S3 bucket region.")
    parser.add_argument("--src-prefix", default="train/", help="Input prefix to scan.")
    parser.add_argument(
        "--dst-prefix",
        default="cleaned/train/",
        help="Prefix for kept images in the same bucket.",
    )
    parser.add_argument(
        "--manifest-prefix",
        default="cleaned/manifests/",
        help="Prefix where run metadata/manifests are written.",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        help="Run folder under manifest-prefix.",
    )
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        help="OpenCLIP model name (e.g., ViT-B-32, ViT-L-14).",
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained weight spec.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Images per inference batch.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="People score cutoff. >= threshold => removed.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional hard limit for scanned images (0 = all).",
    )
    parser.add_argument(
        "--status-file",
        default="",
        help="Optional path to write JSON status updates for monitoring.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify only; do not write cleaned objects.",
    )
    parser.add_argument(
        "--emit-presigned-urls",
        action="store_true",
        help="Add presigned URL field in removed/kept manifests.",
    )
    parser.add_argument(
        "--presign-exp-seconds",
        type=int,
        default=7 * 24 * 3600,
        help="Presigned URL expiry when emitted.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts for individual object operations.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index to process (paired with --shard-count).",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="How many total shards to split the keyspace into.",
    )
    return parser.parse_args()


def s3_https_url(bucket: str, region: str, key: str) -> str:
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def image_id_from_key(key: str) -> str:
    return key.rsplit("/", 1)[-1].split(".")[0]


def is_image_key(key: str) -> bool:
    lower = key.lower()
    return lower.endswith(IMAGE_EXTS)


def iter_candidate_keys(
    s3_client,
    bucket: str,
    prefix: str,
    max_images: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Iterable[Dict]:
    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not is_image_key(key):
                continue
            if shard_count > 1:
                shard = zlib.crc32(key.encode("utf-8")) % shard_count
                if shard != shard_index:
                    continue
            yield obj
            count += 1
            if max_images and count >= max_images:
                return


def make_prompt_vectors(
    model: torch.nn.Module,
    device: torch.device,
    positive_prompts: List[str],
    negative_prompts: List[str],
) -> torch.Tensor:
    with torch.no_grad():
        pos_tokens = open_clip.tokenize(positive_prompts).to(device)
        neg_tokens = open_clip.tokenize(negative_prompts).to(device)

        pos_feat = model.encode_text(pos_tokens)
        neg_feat = model.encode_text(neg_tokens)

        pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
        neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)

        pos_vec = pos_feat.mean(dim=0, keepdim=True)
        neg_vec = neg_feat.mean(dim=0, keepdim=True)
        pos_vec = pos_vec / pos_vec.norm(dim=-1, keepdim=True)
        neg_vec = neg_vec / neg_vec.norm(dim=-1, keepdim=True)
        return torch.cat([pos_vec, neg_vec], dim=0)


def row_from_obj(
    bucket: str,
    region: str,
    obj: Dict,
    people_prob: float = None,
    presigned_url: str = "",
) -> Dict:
    key = obj["Key"]
    return {
        "image_id": image_id_from_key(key),
        "s3_key": key,
        "s3_uri": f"s3://{bucket}/{key}",
        "https_url": s3_https_url(bucket, region, key),
        "presigned_url": presigned_url,
        "people_prob": people_prob,
        "size": obj.get("Size", ""),
        "etag": obj.get("ETag", "").strip('"'),
        "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else "",
    }


def copy_to_cleaned(s3_client, bucket: str, src_key: str, dst_prefix: str, src_prefix: str) -> str:
    if src_key.startswith(src_prefix):
        relative = src_key[len(src_prefix) :]
        dst_key = f"{dst_prefix.rstrip('/')}/{relative}"
    else:
        dst_key = f"{dst_prefix.rstrip('/')}/{src_key}"
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
    )
    return dst_key


def safe_generate_presigned(s3_client, bucket: str, key: str, expires_in: int) -> str:
    try:
        return s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
    except Exception:
        return ""


def upload_text(s3_client, bucket: str, key: str, text: str) -> None:
    s3_client.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))


def dicts_to_csv(rows: List[Dict]) -> str:
    if not rows:
        return "image_id,s3_key,s3_uri,https_url,presigned_url,people_prob,size,etag,last_modified\n"
    keys = ["image_id", "s3_key", "s3_uri", "https_url", "presigned_url", "people_prob", "size", "etag", "last_modified"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        row = {k: row.get(k, "") for k in keys}
        writer.writerow(row)
    return output.getvalue()


def main() -> None:
    args = parse_args()
    status_file = args.status_file.strip()
    monitor_status = len(status_file) > 0

    def write_status(status: str, extra: Dict) -> None:
        if not monitor_status:
            return
        os.makedirs(os.path.dirname(status_file) or ".", exist_ok=True)
        payload = {
            "status": status,
            "run_id": args.run_id,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "bucket": args.bucket,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(extra)
        tmp_path = f"{status_file}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, status_file)

    s3 = boto3.client(
        "s3",
        region_name=args.region,
        config=Config(
            retries={"max_attempts": max(3, args.max_retries + 1), "mode": "standard"}
        ),
    )

    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("--shard-index must be in [0, shard-count).")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    positive_prompts = [
        "a photo of a person",
        "a photo of a human",
        "a photo of people",
        "a photo of a man",
        "a photo of a woman",
        "a photo of a child",
        "a portrait of a person",
    ]
    negative_prompts = [
        "a photo of a landscape",
        "a photo of a building",
        "a photo of food",
        "a photo of an object",
        "an empty room",
    ]
    class_text = make_prompt_vectors(
        model, device, positive_prompts=positive_prompts, negative_prompts=negative_prompts
    )

    shard_suffix = ""
    if args.shard_count > 1:
        shard_suffix = f"/shard-{args.shard_index:04d}-of-{args.shard_count:04d}"
    manifest_base = f"{args.manifest_prefix.rstrip('/')}/{args.run_id}{shard_suffix}"
    removed_rows: List[Dict] = []
    kept_rows: List[Dict] = []
    error_rows: List[Dict] = []

    candidate_count = 0
    removed_count = 0
    kept_count = 0
    error_count = 0
    last_status_update = datetime.now(timezone.utc)

    write_status(
        "starting",
        {
            "model": args.model,
            "pretrained": args.pretrained,
            "threshold": args.threshold,
            "batch_size": args.batch_size,
            "max_images": args.max_images,
        },
    )

    batch: List[torch.Tensor] = []
    batch_meta: List[Dict] = []

    def flush_batch() -> None:
        nonlocal removed_count, kept_count, error_count, last_status_update
        if not batch:
            return

        images = torch.stack(batch, dim=0).to(device)
        with torch.no_grad():
            ctx = torch.autocast("cuda") if device.type == "cuda" else nullcontext()
            with ctx:
                image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.to(class_text.dtype)
            logits = 100.0 * image_features @ class_text.T
            probs = logits.softmax(dim=-1)[:, 0].detach().cpu()

        for meta, score in zip(batch_meta, probs):
            score_float = float(score)
            presigned = (
                safe_generate_presigned(s3, args.bucket, meta["Key"], args.presign_exp_seconds)
                if args.emit_presigned_urls
                else ""
            )
            row = row_from_obj(
                args.bucket, args.region, meta, score_float, presigned
            )
            if score_float >= args.threshold:
                removed_rows.append(row)
                removed_count += 1
            else:
                kept_rows.append(row)
                kept_count += 1
                if not args.dry_run:
                    try:
                        copy_to_cleaned(
                            s3,
                            args.bucket,
                            meta["Key"],
                            args.dst_prefix,
                            args.src_prefix,
                        )
                    except Exception as e:
                        error_count += 1
                        error_rows.append(
                            {
                                "s3_key": meta["Key"],
                                "error": str(e),
                            }
                        )

        batch.clear()
        batch_meta.clear()

        now = datetime.now(timezone.utc)
        elapsed = (now - last_status_update).total_seconds()
        if elapsed >= 2:
            write_status(
                "running",
                {
                    "candidate_count": candidate_count,
                    "removed_count": removed_count,
                    "kept_count": kept_count,
                    "error_count": error_count,
                    "status_note": f"last_flush_at_{now.isoformat()}",
                },
            )
            last_status_update = now

    source_iterator = iter_candidate_keys(
        s3,
        args.bucket,
        args.src_prefix,
        args.max_images,
        args.shard_index,
        args.shard_count,
    )
    pbar = tqdm(desc="Scanning images", unit="img")

    for obj in source_iterator:
        candidate_count += 1
        pbar.update(1)
        if candidate_count % 100 == 0:
            write_status(
                "running",
                {
                    "candidate_count": candidate_count,
                    "removed_count": removed_count,
                    "kept_count": kept_count,
                    "error_count": error_count,
                },
            )
        key = obj["Key"]
        try:
            obj_data = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read()
            image = Image.open(io.BytesIO(obj_data)).convert("RGB")
            batch.append(preprocess(image))
            batch_meta.append(obj)
        except Exception as e:
            error_count += 1
            error_rows.append(
                {"s3_key": key, "error": f"download/decode: {type(e).__name__}: {e}"}
            )
            continue

        if len(batch) >= args.batch_size:
            flush_batch()

    if batch:
        flush_batch()

    pbar.close()

    write_status(
        "writing_outputs",
        {
            "candidate_count": candidate_count,
            "removed_count": removed_count,
            "kept_count": kept_count,
            "error_count": error_count,
        },
    )

    removed_rows.sort(key=lambda r: r["people_prob"], reverse=True)
    kept_rows.sort(key=lambda r: r["people_prob"], reverse=True)

    removed_csv = dicts_to_csv(removed_rows)
    kept_csv = dicts_to_csv(kept_rows)
    removed_ids_csv = "\n".join([r["image_id"] for r in removed_rows]) + (
        "\n" if removed_rows else ""
    )
    removed_urls_csv = "\n".join([r["s3_uri"] for r in removed_rows]) + (
        "\n" if removed_rows else ""
    )

    metadata = {
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "region": args.region,
        "src_prefix": args.src_prefix,
        "dst_prefix": args.dst_prefix,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "manifest_prefix": manifest_base,
        "dry_run": args.dry_run,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "max_images": args.max_images,
        "model": {"name": args.model, "pretrained": args.pretrained},
        "counts": {
            "scanned": candidate_count,
            "removed": removed_count,
            "kept": kept_count,
            "errors": error_count,
        },
        "prompts": {
            "positive": positive_prompts,
            "negative": negative_prompts,
        },
    }

    write_status(
        "finalizing",
        {
            "manifest_prefix": manifest_base,
            "candidate_count": candidate_count,
            "removed_count": removed_count,
            "kept_count": kept_count,
            "error_count": error_count,
        },
    )

    if removed_count + kept_count + error_count != candidate_count and not args.dry_run:
        error_rows.append(
            {
                "s3_key": "__parity_warning__",
                "error": (
                    f"Parity mismatch: scanned={candidate_count} "
                    f"removed+kept+errors={removed_count + kept_count + error_count}"
                ),
            }
        )

    manifest_keys = {
        "removed_csv": f"{manifest_base}/removed.csv",
        "kept_csv": f"{manifest_base}/kept.csv",
        "removed_image_ids": f"{manifest_base}/removed_image_ids.txt",
        "removed_s3_urls": f"{manifest_base}/removed_s3_urls.txt",
        "errors_csv": f"{manifest_base}/errors.csv",
        "metadata": f"{manifest_base}/run_metadata.json",
    }

    upload_text(s3, args.bucket, manifest_keys["removed_csv"], removed_csv)
    upload_text(s3, args.bucket, manifest_keys["kept_csv"], kept_csv)
    upload_text(s3, args.bucket, manifest_keys["removed_image_ids"], removed_ids_csv)
    upload_text(s3, args.bucket, manifest_keys["removed_s3_urls"], removed_urls_csv)
    upload_text(
        s3,
        args.bucket,
        manifest_keys["errors_csv"],
        dicts_to_csv(error_rows)
        if error_rows
        else "s3_key,error\n",
    )
    upload_text(s3, args.bucket, manifest_keys["metadata"], json.dumps(metadata, indent=2))

    print(json.dumps(
        {
            "status": "complete",
            "manifest_prefix": manifest_base,
            "candidate_count": candidate_count,
            "removed_count": removed_count,
            "kept_count": kept_count,
            "error_count": error_count,
            "outputs": manifest_keys,
            "cleaned_prefix": f"s3://{args.bucket}/{args.dst_prefix}",
        },
        indent=2,
    ))
    write_status(
        "complete",
        {
            "manifest_prefix": manifest_base,
            "candidate_count": candidate_count,
            "removed_count": removed_count,
            "kept_count": kept_count,
            "error_count": error_count,
            "completed": True,
        },
    )


if __name__ == "__main__":
    main()
