#!/usr/bin/env python3
"""Merge per-shard CLIP removal manifests into one consolidated manifest."""

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from typing import Dict, List

import boto3


REMOVED_CSV_FIELDS = [
    "image_id",
    "s3_key",
    "s3_uri",
    "https_url",
    "presigned_url",
    "people_prob",
    "size",
    "etag",
    "last_modified",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge shard removal manifests.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name.")
    parser.add_argument("--region", required=True, help="S3 region.")
    parser.add_argument("--run-id", required=True, help="Run id used by cleaning pass.")
    parser.add_argument(
        "--manifest-prefix",
        default="cleaned/manifests/",
        help="Prefix where shard runs were written.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=8,
        help="Number of shards used in cleaning run.",
    )
    parser.add_argument(
        "--output-suffix",
        default="aggregate",
        help="Folder name for merged outputs within the run folder.",
    )
    return parser.parse_args()


def read_csv_rows(s3_client, bucket: str, key: str) -> List[Dict]:
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    rows: List[Dict] = []
    if not body.strip():
        return rows
    reader = csv.DictReader(io.StringIO(body))
    for row in reader:
        rows.append(row)
    return rows


def read_lines(s3_client, bucket: str, key: str) -> List[str]:
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    return [line.strip() for line in body.splitlines() if line.strip()]


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def dicts_to_csv(rows: List[Dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=REMOVED_CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in REMOVED_CSV_FIELDS})
    return output.getvalue()


def rows_to_text_lines(rows: List[str]) -> str:
    return "\n".join(unique_preserve_order(rows)) + ("\n" if rows else "")


def main() -> None:
    args = parse_args()

    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")

    s3 = boto3.client("s3", region_name=args.region)
    run_root = f"{args.manifest_prefix.rstrip('/')}/{args.run_id}"
    merged_key_prefix = f"{run_root}/{args.output_suffix}"

    removed_rows: List[Dict] = []
    removed_ids: List[str] = []
    removed_urls: List[str] = []
    shard_status = []

    for shard_idx in range(args.shard_count):
        shard_name = f"shard-{shard_idx:04d}-of-{args.shard_count:04d}"
        shard_base = f"{run_root}/{shard_name}"
        shard_report = {
            "shard": shard_name,
            "found": True,
            "counts": {"removed_rows": 0, "removed_ids": 0, "removed_urls": 0},
            "errors": [],
        }

        removed_csv_key = f"{shard_base}/removed.csv"
        removed_ids_key = f"{shard_base}/removed_image_ids.txt"
        removed_urls_key = f"{shard_base}/removed_s3_urls.txt"

        try:
            rows = read_csv_rows(s3, args.bucket, removed_csv_key)
            shard_report["counts"]["removed_rows"] = len(rows)
            removed_rows.extend(rows)
        except Exception as e:
            shard_report["found"] = False
            shard_report["errors"].append(f"missing {removed_csv_key}: {e}")

        try:
            ids = read_lines(s3, args.bucket, removed_ids_key)
            shard_report["counts"]["removed_ids"] = len(ids)
            removed_ids.extend(ids)
        except Exception as e:
            shard_report["errors"].append(f"missing {removed_ids_key}: {e}")

        try:
            urls = read_lines(s3, args.bucket, removed_urls_key)
            shard_report["counts"]["removed_urls"] = len(urls)
            removed_urls.extend(urls)
        except Exception as e:
            shard_report["errors"].append(f"missing {removed_urls_key}: {e}")

        shard_status.append(shard_report)

    # Deduplicate to be safe in case shards overlap.
    unique_rows = {}
    for row in removed_rows:
        key = row.get("s3_key", "").strip()
        if key and key not in unique_rows:
            unique_rows[key] = row

    unique_ids = unique_preserve_order(removed_ids)
    unique_urls = unique_preserve_order(removed_urls)

    merged_rows = list(unique_rows.values())
    merged_removed_csv = dicts_to_csv(merged_rows)
    merged_removed_ids = rows_to_text_lines(unique_ids)
    merged_removed_urls = rows_to_text_lines(unique_urls)

    metadata = {
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "region": args.region,
        "manifest_root": run_root,
        "output_prefix": merged_key_prefix,
        "counts": {
            "shard_count": args.shard_count,
            "merged_removed_rows": len(merged_rows),
            "merged_removed_ids": len(unique_ids),
            "merged_removed_urls": len(unique_urls),
        },
        "shard_status": shard_status,
    }

    s3.put_object(
        Bucket=args.bucket,
        Key=f"{merged_key_prefix}/removed_all.csv",
        Body=merged_removed_csv.encode("utf-8"),
    )
    s3.put_object(
        Bucket=args.bucket,
        Key=f"{merged_key_prefix}/removed_all_image_ids.txt",
        Body=merged_removed_ids.encode("utf-8"),
    )
    s3.put_object(
        Bucket=args.bucket,
        Key=f"{merged_key_prefix}/removed_all_s3_urls.txt",
        Body=merged_removed_urls.encode("utf-8"),
    )
    s3.put_object(
        Bucket=args.bucket,
        Key=f"{merged_key_prefix}/merge_metadata.json",
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
