# Databricks notebook source
# MAGIC %pip install --upgrade transformers huggingface_hub
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 1
import pandas as pd
from random import randrange

# Load Dolly dataset directly from HuggingFace as parquet
url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
dataset = pd.read_json(url, lines=True)

print(f"dataset size: {len(dataset)}")
print(dataset.iloc[randrange(len(dataset))].to_dict())

# COMMAND ----------

display(dataset)

# COMMAND ----------

def format_dolly(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample['context']) > 0 else None
    response = f"### Answer\n{sample['response']}"
    prompt = "\n\n".join([i for i in [instruction,context,response] if i is not None ])
    return prompt

# COMMAND ----------

# DBTITLE 1,Cell 3
print(format_dolly(dataset.iloc[1]))

# COMMAND ----------

import os

os.environ["HF_TOKEN"] = "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"

# COMMAND ----------

# DBTITLE 1,Cell 6


# COMMAND ----------

from transformers import AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# COMMAND ----------

from random import randint
from itertools import chain
from functools import partial
def template_dataset(sample):
    print(sample)
    sample['text'] = f"{format_dolly(sample)}{tokenizer.eos_token}"
    return sample

# COMMAND ----------


dataset = dataset.apply(template_dataset, axis=1)

# COMMAND ----------

display(dataset)

# COMMAND ----------

#print(dataset[randint(0,len(dataset))]['text'])

remainder = {'input_ids':[],'attention_mask':[],'token_type_ids':[]}

# COMMAND ----------

def chunk(sample, chunk_length = 2048):

    global remainder

    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}

    concatenated_examples = {
        k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()
    }

    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    if batch_total_length >= chunk_length:
        batch_total_length = (batch_total_length // chunk_length) * chunk_length
        
    result = {
        k: [t[i: i + chunk_length] for i in range(0, batch_total_length, chunk_length)] for k,t in concatenated_examples.items()
    }
    
    remainder = {
        k: concatenated_examples[k][batch_total_length:] for k in concatenated_examples.keys()
    }

    result["labels"] = result["input_ids"].copy()
    return result

# COMMAND ----------

lm_dataset = dataset.apply(lambda row: tokenizer(row["text"]), axis=1, result_type='expand')

# COMMAND ----------

# DBTITLE 1,Cell 14
from itertools import chain
from functools import partial

# Tokenize all samples first
tokenized_data = []
for idx, row in dataset.iterrows():
    tokens = tokenizer(row["text"])
    tokenized_data.append(tokens)

# Now implement the chunking logic from the original code
remainder = {'input_ids': [], 'attention_mask': []}
chunked_samples = []

def chunk_batch(tokenized_list, chunk_length=2048):
    global remainder
    
    # Concatenate all token sequences from this batch
    all_input_ids = remainder['input_ids'].copy()
    all_attention_masks = remainder['attention_mask'].copy()
    
    for tokens in tokenized_list:
        all_input_ids.extend(tokens['input_ids'])
        all_attention_masks.extend(tokens['attention_mask'])
    
    # Calculate how many complete chunks we can make
    total_length = len(all_input_ids)
    
    if total_length >= chunk_length:
        # Number of complete chunks
        num_chunks = total_length // chunk_length
        usable_length = num_chunks * chunk_length
        
        # Create chunks
        for i in range(0, usable_length, chunk_length):
            chunked_samples.append({
                'input_ids': all_input_ids[i:i + chunk_length],
                'attention_mask': all_attention_masks[i:i + chunk_length],
                'labels': all_input_ids[i:i + chunk_length]  # For LM, labels = input_ids
            })
        
        # Save remainder
        remainder['input_ids'] = all_input_ids[usable_length:]
        remainder['attention_mask'] = all_attention_masks[usable_length:]
    else:
        # Not enough for even one chunk, save as remainder
        remainder['input_ids'] = all_input_ids
        remainder['attention_mask'] = all_attention_masks

# Process all tokenized data
chunk_batch(tokenized_data, chunk_length=2048)

# Convert to DataFrame
lm_dataset = pd.DataFrame(chunked_samples)

print(f"Total number of samples: {len(lm_dataset)}")  # you have this many chunks, and each is 2048 tokens long