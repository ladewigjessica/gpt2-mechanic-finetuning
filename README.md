# GPT-2 Fine-Tuning on a Car Mechanic Corpus 🔧

Fine-tuning a pre-trained GPT-2 model on a custom domain corpus, built as a course assignment for the Python for AI and LLMs programme at CINEL.

The corpus was created specifically for this project — 100 client/mechanic dialogue pairs covering common vehicle symptoms and diagnoses. The goal was to adapt GPT-2's language generation to this domain: given a client description of a problem, the model should generate a plausible mechanic response.

## What it does

- Extracts text from a PDF corpus using `pypdf`
- Loads and tokenises the corpus with the Hugging Face `datasets` library
- Groups tokens into 1024-token blocks for causal language modelling
- Fine-tunes GPT-2 using the Hugging Face `Trainer`
- Generates text with the fine-tuned model given a client prompt

## Notes

The corpus is small (100 dialogue pairs, 5 training blocks), so the generated output is not coherent — the model does not have enough data to learn the pattern reliably. The project demonstrates the full fine-tuning pipeline rather than a production-ready model. A larger corpus would produce meaningfully better results.

## Tools

Python · PyTorch · Hugging Face Transformers · Hugging Face Datasets · pypdf · Google Colab (T4 GPU)

## Files

- `gpt2_mechanic_finetuning.ipynb` — main notebook with full pipeline and outputs
- `mechanic_corpus.pdf` — custom dialogue corpus used for training
