"""Upload only the README to HuggingFace dataset."""

from huggingface_hub import HfApi
import os

def upload_readme_only(repo_id, readme_path=None):
    """
    Upload README.md to HuggingFace dataset repo without touching data.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "CoffeeGitta/pika-math-generations")
        readme_path: Path to README file (defaults to DATASET_README_TEMPLATE.md)
    """
    api = HfApi()
    
    if readme_path is None:
        readme_path = "scripts/DATASET_README_TEMPLATE.md"
    
    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"README not found at {readme_path}")
    
    with open(readme_path, 'r') as f:
        readme_content = f.read()
    
    # Upload to HuggingFace
    api.upload_file(
        path_or_fileobj=readme_content.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    print(f"✅ Uploaded README to {repo_id}")


if __name__ == "__main__":
    REPO_ID = "CoffeeGitta/pika-math-generations"
    
    upload_readme_only(repo_id=REPO_ID)
