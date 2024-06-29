from collections import Counter
from datasets import load_dataset # huggingface datasets
import json
import pandas as pd
from transformers import AutoTokenizer

# Adapted from Andrej Karpathy's nanoGPT repo:
# https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
def count_tokens_in_openwebtext(
    tokenizer_model_id,
    split="train",
    num_load_workers=8,
    num_tokenize_workers=8,
    return_df=True,
    save_to=""
):
    """
    Loads the openwebtext dataset, tokenizes it, and then counts the number of
    occurrences of each token

    tokenizer_model_id = Hugging Face model ID for the tokenizer to use
    split = Which split of openwebtext to count ("train" or "val")
    num_load_workers = Number of workers in load_dataset() call; best number
        might be different from num_tokenize_workers below as it also depends on
        NW speed; it is better than 1 usually though
    num_tokenize_workers = number of workers in .map() call; good number to use
        is ~order number of cpu cores // 2
    return_df = (bool) Whether to return a DataFrame (True) or a Counter (False)
    save_to = (str) Optional filepath of where to save the final return object
        from return_df as a JSON file. If not provided, will not save the
        results to disk.
    """
    assert split in ("train", "val")
    assert num_load_workers >= 1 and num_tokenize_workers >= 1

    # Takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset(
        "openwebtext",
        num_proc=num_load_workers,
        # "The repository for openwebtext contains custom code which must be
        # executed to correctly load the dataset"
        trust_remote_code=True
    )

    # openwebtext by default only contains a train split, so create a test split
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005,
        seed=2357,  # For consistency of results, using same seed from nanoGPT
        shuffle=True
    )
    # Rename the test split to val
    split_dataset['val'] = split_dataset.pop('test')

    # Prep for tokenization
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)

    def tokenize(example):
        ids = tokenizer.encode(example["text"])
        ids.append(tokenizer.eos_token_id) # add the end of text token
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Actually tokenize the dataset
    tokenized = split_dataset.map(
        tokenize,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_tokenize_workers,
    )
    tokenized

    # Count the tokens
    counter = Counter()
    for text in tokenized[split]:
        counter.update(text["ids"])

    if return_df:
        vocab_size = len(tokenizer)

        df = pd.DataFrame(dict(
            token_id=list(range(vocab_size)),
            token=tokenizer.batch_decode(
                list(range(vocab_size)),
                clean_up_tokenization_spaces=False
            ),
            n=[counter[i] for i in range(vocab_size)]
        ))
        
        if save_to:
            df.to_json(save_to, index=False)

        return df
    else:
        if save_to:
            with open(save_to, "w") as f:
                json.dump(counter, f)

        return counter
