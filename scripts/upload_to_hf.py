#!/usr/bin/env python3
"""
Upload prepared probes to HuggingFace Hub.

Prerequisites:
    1. Run prepare_hf_probes.py first to create clean probe folder
    2. Login with: huggingface-cli login

Usage:
    python scripts/upload_to_hf.py --repo-id CoffeeGitta/pika-probes
    python scripts/upload_to_hf.py --repo-id CoffeeGitta/pika-probes --private
"""

import argparse
from pathlib import Path

# Default repository
DEFAULT_REPO_ID = "CoffeeGitta/pika-probes"


def main():
    parser = argparse.ArgumentParser(description="Upload probes to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID,
                        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--probes-dir", type=Path, default=Path("probes_for_hf"),
                        help="Directory containing prepared probes")
    parser.add_argument("--private", action="store_true",
                        help="Make the repository private")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without actually uploading")
    
    args = parser.parse_args()
    
    if not args.probes_dir.exists():
        print(f"❌ Probes directory not found: {args.probes_dir}")
        print("Run prepare_hf_probes.py first to create it.")
        return 1
    
    # Check for required files
    readme = args.probes_dir / "README.md"
    manifest = args.probes_dir / "manifest.json"
    
    if not readme.exists():
        print(f"❌ README.md not found in {args.probes_dir}")
        return 1
    
    # Count probes
    probe_dirs = [d for d in args.probes_dir.iterdir() if d.is_dir()]
    print(f"Found {len(probe_dirs)} probe folders to upload")
    
    if args.dry_run:
        print(f"\nDry run - would upload to: {args.repo_id}")
        print(f"  README.md")
        if manifest.exists():
            print(f"  manifest.json")
        for d in sorted(probe_dirs):
            files = list(d.iterdir())
            print(f"  {d.name}/ ({len(files)} files)")
        return 0
    
    # Import huggingface_hub
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        return 1
    
    api = HfApi()
    
    # Check authentication
    try:
        user_info = api.whoami()
        print(f"✓ Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Not authenticated. Run: huggingface-cli login")
        print(f"   Error: {e}")
        return 1
    
    # Create or get repo
    print(f"\nCreating/accessing repo: {args.repo_id}")
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"✓ Repository ready: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"❌ Failed to create repo: {e}")
        return 1
    
    # Upload the entire folder
    print(f"\nUploading {args.probes_dir} to {args.repo_id}...")
    try:
        api.upload_folder(
            folder_path=str(args.probes_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload PIKA probes",
        )
        print(f"\n✅ Successfully uploaded to: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
