# OpenWebText Token Counting

## Tokenizers / Models Already Counted

This folder already includes the OpenWebText train token counts for the following tokenizers / model families:
 - [Gemma](openwebtext_train_gemma_token_counts.json)
 - [GPT 2](openwebtext_train_gpt2_token_counts.json)
 - [Llama 3](openwebtext_train_llama3_token_counts.json)
 - [Mistral](openwebtext_train_mistral_token_counts.json)
 - [Mixtral](openwebtext_train_mixtral_token_counts.json)
 - [Phi 3 (Mini and Medium)](openwebtext_train_phi3_token_counts.json)
 - [Phi 3 (Small)](openwebtext_train_phi3-small_token_counts.json)

These are saved as JSON serialized `pandas` DataFrames and can be loaded as follows:

```python
import pandas as pd

token_counts_df = pd.read_json("your/path/to/example_token_counts.json")
```

## Counting for a New Tokenizer / Model

If you want to count the OpenWebText train tokens for a new tokenizer / model, you can use the `count_tokens_in_openwebtext()` function in [counting.py](counting.py). An example of using this function is shown in [Usage_Example.ipynb](Usage_Example.ipynb).

The following packages are required. You can install them normally, if you don't already have them installed.
 - `datasets` (Hugging Face datasets)
 - `pandas`
 - `transformers` (Hugging Face transformers)

The most important parameter is the `tokenizer_model_id` which is the Hugging Face model ID (e.g. `"gpt2"` or `"meta-llama/Meta-Llama-3-70B"`) of the model / tokenizer you're interested in. I also highly recommend setting `save_to` to some filepath so that your counting results will be saved in order to avoid needing to re-run the very slow counting process.
