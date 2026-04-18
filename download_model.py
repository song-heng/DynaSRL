import argparse
import os
from pathlib import Path

# Set a mirror by default for users in China before importing huggingface_hub.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import login, snapshot_download

DEFAULT_MODELS = [
    "Qwen/Qwen3-8B",
]


def normalize_model_name(repo_id):
    model_name = repo_id.split("/")[-1]
    if model_name.startswith("Meta-"):
        return model_name[5:]
    return model_name


def parse_args():
    parser = argparse.ArgumentParser(description="Download base models used by DynaSRL.")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated Hugging Face repo IDs.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/root/autodl-tmp/models",
        help="Directory for downloaded models.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to the HF_TOKEN environment variable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_ids = [repo_id.strip() for repo_id in args.models.split(",") if repo_id.strip()]
    if not repo_ids:
        raise ValueError("No model repo IDs were provided. Use --models to specify at least one repo.")

    token = args.token.strip() if args.token else None
    if token:
        login(token=token, add_to_git_credential=False)

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for repo_id in repo_ids:
        model_dir = target_dir / normalize_model_name(repo_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 60)
        print(f"Starting download for: {repo_id}")
        print(f"Target path: {model_dir}")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_dir),
                token=token,
            )
            print(f"Download successfully completed for {repo_id}!")
        except Exception as exc:
            print(f"Download failed for {repo_id}. Error details:\n{exc}")

    print("-" * 60)
    print("All download tasks finished!")


if __name__ == "__main__":
    main()
