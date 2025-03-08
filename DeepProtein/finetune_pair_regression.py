print("yes")
import os
import sys

import torch.nn as nn

from transformers import (
    T5Config,
    T5EncoderModel,
    T5Tokenizer,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
print("yes")

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from DeepProtein.dataset import *
from datasets import Dataset
print("yes")

from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Develop

data = Develop(name='SAbDab_Chen')
split = data.get_split()

train_antibody_1, train_antibody_2 = to_two_seq(split, 'train', 'Antibody', sep=",")
valid_antibody_1, valid_antibody_2 = to_two_seq(split, 'valid', 'Antibody', sep=",")

y_train, y_valid, y_test = split['train']['Y'], split['valid']['Y'], split['test']['Y']

train= list(zip(train_antibody_1, train_antibody_2, y_train))
valid = list(zip(valid_antibody_1, valid_antibody_2, y_valid))


print("yes")


class T5PairRegressionModel(PreTrainedModel):
    config_class = T5Config

    def __init__(self, config, d_model=None):
        super().__init__(config)
        self.encoder = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        hidden_dim = d_model if d_model is not None else config.d_model
        self.regression_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)
        pooled_output = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        logits = self.regression_head(pooled_output).squeeze(-1)  # [batch_size]

        loss = None
        if labels is not None:
            labels = labels.to(torch.bfloat16)
            loss = nn.MSELoss()(logits, labels)

        return {"loss": loss, "logits": logits}


def build_fluo_data():
    data_list = []
    for i in range(len(train)):
        seq1, seq2, lab = train[i]
        data_list.append({
            "sequence1": seq1,
            "sequence2": seq2,
            "label": float(lab),
        })
    return data_list

def build_fluo_valid():
    data_list = []
    for i in range(len(valid)):
        seq1, seq2, lab = valid[i]
        data_list.append({
            "sequence1": seq1,
            "sequence2": seq2,
            "label": float(lab),
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
        seq1_list = [" ".join(list(re.sub(r"[UZOB]", "X", s1[:200]))) for s1 in examples["sequence1"]]
        seq2_list = [" ".join(list(re.sub(r"[UZOB]", "X", s2[:200]))) for s2 in examples["sequence2"]]
        out = tokenizer(
            seq1_list,
            seq2_list,
            padding=True,
            truncation=True,
            max_length=800,
            return_tensors="pt"
        )
        out["labels"] = examples["label"]
        return out

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")


    data_collator = DataCollatorWithPadding(tokenizer)


    config = T5Config.from_pretrained(MODEL_NAME)
    model = T5PairRegressionModel.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=torch.bfloat16
    )


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        labels = labels.flatten()
        mse = ((predictions - labels) ** 2).mean().item()
        return {"mse": mse}

    training_args = TrainingArguments(
        output_dir="/cluster/scratch/jiaxie/checkpoints",
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=200,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        learning_rate=5e-4,
        weight_decay=0.0,
        logging_steps=10,
        gradient_accumulation_steps=1,
        load_best_model_at_end=True,
        bf16=True,
        fp16=False,
        seed=42,
        lr_scheduler_type="constant",
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


    save_path = "/cluster/scratch/jiaxie/.cache/DeepProtT5-SAbDab-Chen"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    from huggingface_hub import upload_folder
    upload_folder(
        folder_path=save_path,
        repo_id="jiaxie/DeepProtT5-SAbDab-Chen",
        repo_type="model"
    )

if __name__ == "__main__":
    main()
