import os

from cs336_basics.tokenizer.bpe_tokenizer import (
    train_bpe,
    load_tokenizer_from_dir,
)

TINY_STORIES = {
    "train_data_path": "data/TinyStoriesV2-GPT4-train.txt",
    "dev_data_path": "data/TinyStoriesV2-GPT4-valid.txt",
    "vocab_size": 10_000,
    "special_tokens": ["<|endoftext|>"],
    "save_dir": "./datasets/tiny_stories",
}

OWT = {
    "train_data_path": "data/owt_train.txt",
    "dev_data_path": "data/owt_valid.txt",
    "vocab_size": 32_000,
    "special_tokens":[],
    "save_dir":"./datasets/owt",
}

if __name__ == "__main__":
    dataset = OWT

    if not os.path.exists(dataset["save_dir"]):
        os.makedirs(dataset["save_dir"])
        train_bpe(
            dataset["train_data_path"],
            vocab_size=dataset["vocab_size"],
            special_tokens=dataset["special_tokens"],
            verbose=True,
            save_path=dataset["save_dir"],
        )

        print(f"BPE tokenizer trained and saved to {dataset['save_dir']}")
    elif os.path.exists(os.path.join(dataset["save_dir"], "vocab.json")) and os.path.exists(
        os.path.join(dataset["save_dir"], "merges.txt")
    ):
        print(f"BPE tokenizer already exists in {dataset['save_dir']}")
    else:
        print(f"Save directory {dataset['save_dir']} exists but tokenizer files not found.")
        train_bpe(
            dataset["train_data_path"],
            vocab_size=dataset["vocab_size"],
            special_tokens=dataset["special_tokens"],
            verbose=True,
            save_path=dataset["save_dir"],
        )

    tokenizer = load_tokenizer_from_dir(dataset["save_dir"])



