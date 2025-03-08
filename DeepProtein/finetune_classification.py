import os
import sys
import numpy as np
import re
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from transformers import (
    T5Config,
    T5EncoderModel,
    T5Tokenizer,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)



module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from DeepProtein.dataset import *
from datasets import Dataset

train = Solubility(os.getcwd() + '/DeepProtein/data', split='train')
valid = Solubility(os.getcwd() + '/DeepProtein/data', split='valid')



class T5ClassificationModel(PreTrainedModel):
    config_class = T5Config

    def __init__(self, config, d_model=None, num_classes=10):
        super().__init__(config)
        self.num_classes = num_classes

        self.encoder = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

        hidden_dim = d_model if d_model is not None else config.d_model
        self.classification_head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)
        pooled_output = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        logits = self.classification_head(pooled_output)  # [batch_size, num_classes]

        loss = None
        if labels is not None:
            labels = labels.long()
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {
            "loss": loss,
            "logits": logits
        }



def build_fluo_data():
    data_list = []
    for i in range(len(train)):
        seq, lab = train[i]
        lab = int(lab)
        data_list.append({
            "sequence": seq,
            "label": lab,
        })
    return data_list

def build_fluo_valid():
    data_list = []
    for i in range(len(valid)):
        seq, lab = valid[i]
        lab = int(lab)
        data_list.append({
            "sequence": seq,
            "label": lab,
        })
    return data_list


def main():
    fluo_data_list = build_fluo_data()
    fluo_valid_data_list = build_fluo_valid()

    train_dataset = Dataset.from_list(fluo_data_list)
    val_dataset = Dataset.from_list(fluo_valid_data_list)

    MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, do_lower_case=False)

    def tokenize_fn(examples):
        seq_list = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in examples["sequence"]]
        out = tokenizer(seq_list, padding=True, truncation=True, max_length=300, return_tensors="pt")
        out["labels"] = examples["label"]
        return out

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer)


    config = T5Config.from_pretrained(MODEL_NAME)
    num_classes = 2

    model = T5ClassificationModel.from_pretrained(
        MODEL_NAME,
        config=config,
        num_classes=num_classes,
        torch_dtype=torch.bfloat16
    )


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir="/cluster/scratch/jiaxie/checkpoints",
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=30,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        learning_rate=1e-4,
        weight_decay=0.0,
        logging_steps=10,
        gradient_accumulation_steps=1,
        load_best_model_at_end=True,
        bf16=True,
        fp16=False,
        seed=42,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


    save_path = "/cluster/scratch/jiaxie/.cache/DeepProtT5-Solubility"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    from huggingface_hub import upload_folder

    local_folder = "/cluster/scratch/jiaxie/.cache/DeepProtT5-Solubility"
    repo_id = "jiaxie/DeepProtT5-SubCellular-Solubility"

    upload_folder(
        folder_path=local_folder,
        repo_id=repo_id,
        repo_type="model"
    )

if __name__ == "__main__":
    main()
