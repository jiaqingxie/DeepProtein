import os, sys
import torch
import torch.nn as nn

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from DeepProtein.dataset import *

from transformers import (
    T5Config,
    T5EncoderModel,
    PreTrainedModel,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.model_selection import train_test_split
from datasets import Dataset

import numpy as np

# ==========================
# 1) 加载 9 个 DataSet 类
# ==========================
# 单序列回归 (3)
train_fluo = FluorescenceDataset(os.getcwd() + '/DeepProtein/data', 'train')
train_beta = Beta_lactamase(os.getcwd() + '/DeepProtein/data', 'train')
train_stab = Stability(os.getcwd() + '/DeepProtein/data', 'train')

# 单序列分类
#   - "solu" -> 二分类
#   - "sub"  -> 十分类
#   - "sub_bin" -> 二分类
train_solu = Solubility(os.getcwd() + '/DeepProtein/data', 'train')
train_sub = Subcellular(os.getcwd() + '/DeepProtein/data', 'train')  # 10 类
train_sub_bin = BinarySubcellular(os.getcwd() + '/DeepProtein/data', 'train')  # 2 类

# 双序列二分类
train_yeast = Yeast_PPI(os.getcwd() + '/DeepProtein/data', 'train')  # 2 类
train_human = HUMAN_PPI(os.getcwd() + '/DeepProtein/data', 'train')  # 2 类

# 双序列回归
train_aff = PPI_Affinity(os.getcwd() + '/DeepProtein/data', 'train')

# ===========================================
# 2) 自定义 T5 Multi-Head 模型
# ===========================================
import torch
import torch.nn as nn
from transformers import PreTrainedModel, T5Config, T5EncoderModel


class T5MultiHeadModel(PreTrainedModel):
    config_class = T5Config

    def __init__(self, config, num_classes_binary=2, num_classes_10=10, d_model=None):
        super().__init__(config)

        # 1️⃣ T5 Encoder 共享特征提取
        self.encoder = T5EncoderModel(config)
        hidden_dim = d_model if d_model is not None else config.d_model

        # 2️⃣ 三个任务头
        self.classification_head_2 = nn.Linear(hidden_dim, num_classes_binary)
        self.classification_head_10 = nn.Linear(hidden_dim, num_classes_10)
        self.regression_head = nn.Linear(hidden_dim, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, task_head_type=None, labels=None, **kwargs):
        """
        Args:
          - task_head_type: 字符串列表或张量, {"binary","tenclass","regression"}
          - labels: 分类 => int, 回归 => float
        Returns:
          {
            "loss":   标量 or None,
            "logits_binary":     [N_b,2] or None,
            "logits_tenclass":   [N_t,10] or None,
            "logits_regression": [N_r] or None
          }
        """

        # 1️⃣ T5 编码器
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

        # 2️⃣ 取第一个 token 作为 pooled representation
        pooled_output = hidden_states[:, 0, :]  # [batch_size, hidden_dim]

        # 3️⃣ 处理 `task_head_type`
        if isinstance(task_head_type, list):
            task_codes = torch.tensor(
                [0 if x == "binary" else 1 if x == "tenclass" else 2 for x in task_head_type],
                device=pooled_output.device
            )
        else:
            task_codes = task_head_type  # 直接使用张量

        binary_mask = (task_codes == 0)
        tenclass_mask = (task_codes == 1)
        regress_mask = (task_codes == 2)


        loss = None  # ✅ 这里确保 `loss` 正确初始化

        # ========== 处理二分类任务 ==========
        if binary_mask.any():
            idx_b = binary_mask.nonzero(as_tuple=True)[0]
            pooled_b = pooled_output[idx_b]
            logits_binary = self.classification_head_2(pooled_b)  # [Nb,2]

            if labels is not None:
                b_labels = labels[idx_b].long()
                if len(b_labels) > 0:
                    loss_b = nn.CrossEntropyLoss()(logits_binary, b_labels)
                    loss = loss_b if loss is None else (loss + loss_b)

        # ========== 处理十分类任务 ==========
        if tenclass_mask.any():
            idx_t = tenclass_mask.nonzero(as_tuple=True)[0]
            pooled_t = pooled_output[idx_t]
            logits_tenclass = self.classification_head_10(pooled_t)  # [Nt,10]

            if labels is not None:
                t_labels = labels[idx_t].long()
                if len(t_labels) > 0:
                    loss_t = nn.CrossEntropyLoss()(logits_tenclass, t_labels)
                    loss = loss_t if loss is None else (loss + loss_t)

        # ========== 处理回归任务 ==========
        if regress_mask.any():
            idx_r = regress_mask.nonzero(as_tuple=True)[0]
            pooled_r = pooled_output[idx_r]
            logits_regression = self.regression_head(pooled_r).squeeze(-1)  # [Nr]

            if labels is not None:
                r_labels = labels[idx_r].float()

                # 🚀 修正: 避免 `NaN` loss
                if torch.isnan(r_labels).any():
                    print("⚠️ 发现 NaN label，修正为 0!")
                    r_labels = torch.nan_to_num(r_labels, nan=0.0)

                if len(r_labels) > 0:
                    loss_r = nn.MSELoss()(logits_regression, r_labels)
                    loss = loss_r if loss is None else (loss + loss_r)

        # 🚀 修正: 预防 `loss=0` 和 `grad_norm=NaN`
        if loss is None:
            loss = torch.tensor(0.0, device=pooled_output.device)

        return {
            "loss": loss,
        }


# ===========================================
# 3) 构建多任务 data_list
# ===========================================
def build_multi_head_data():
    data_list = []

    # 单序列回归
    def add_single_regression(dataset):
        for i in range(len(dataset)):
            seq, lab = dataset[i]
            data_list.append({
                "task_head_type": "regression",
                "sequence": seq,
                "label": lab
            })

    # 单序列二分类
    def add_single_binary(dataset):
        for i in range(len(dataset)):
            seq, lab = dataset[i]
            data_list.append({
                "task_head_type": "binary",
                "sequence": seq,
                "label": lab
            })

    # 单序列十分类
    def add_single_tenclass(dataset):
        for i in range(len(dataset)):
            seq, lab = dataset[i]
            data_list.append({
                "task_head_type": "tenclass",
                "sequence": seq,
                "label": lab
            })

    # 双序列二分类
    def add_pair_binary(dataset):
        for i in range(len(dataset)):
            s1, s2, lab = dataset[i]
            data_list.append({
                "task_head_type": "binary",
                "sequence": s1 + " <sep> " + s2,
                "label": lab
            })

    # 双序列回归
    def add_pair_regression(dataset):
        for i in range(len(dataset)):
            s1, s2, lab = dataset[i]
            data_list.append({
                "task_head_type": "regression",
                "sequence": s1 + " <sep> " + s2,
                "label": lab
            })

    # === 回归 (3) ===
    add_single_regression(train_fluo)
    add_single_regression(train_beta)
    add_single_regression(train_stab)

    # === 单序列分类 ===
    # solu => 2-class
    add_single_binary(train_solu)
    # sub => 10-class
    add_single_tenclass(train_sub)
    # sub_bin => 2-class
    add_single_binary(train_sub_bin)

    # === 双序列二分类 ===
    # add_pair_binary(train_yeast)
    # add_pair_binary(train_human)
    #
    # # === 双序列回归 ===
    # add_pair_regression(train_aff)

    return data_list

# ===========================================
# 4) 主函数
# ===========================================
def main():
    # 1) 构建多任务数据
    all_data_list = build_multi_head_data()

    # 2) 统一 label 的类型，以免混合数组和标量
    for ex in all_data_list:
        lab = ex["label"]
        head_type = ex["task_head_type"]
        # 如果 label 是 numpy 数组，就先转成标量或 list
        if isinstance(lab, np.ndarray):
            if lab.size == 1:
                # 如果是一维单值数组，直接取出来
                lab = lab.item()  # 变为纯 Python 标量
            else:
                # 如果是多维数组，看需要多标签或只取第一项？
                # 暂时演示：直接取 list => 这会变成 list，可能仍然混类型
                # 如果你确实需要多维标签，请保证所有样本都统一成 list
                lab = lab.tolist()

        # 对分类任务 => int，对回归任务 => float
        if head_type in ["binary", "tenclass"]:
            # 二分类或十分类 => int
            ex["label"] = int(lab)
        else:
            # 回归 => float
            ex["label"] = float(lab)

    # 3) 转为 HF Dataset
    from collections import defaultdict
    def inspect_data_types(data_list):
        key_types = defaultdict(set)
        for i, example in enumerate(data_list):
            for k, v in example.items():
                key_types[k].add(type(v))
        for k, types_found in key_types.items():
            print(f"{k}: {types_found}")

    print("=== 检查转换后 data_list 的类型分布 ===")
    inspect_data_types(all_data_list)

    full_dataset = Dataset.from_list(all_data_list)

    # 4) Train/Val 切分
    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]




    # 5) Tokenizer
    MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    task_mapping = {"binary": 0, "tenclass": 1, "regression": 2}

    def tokenize_fn(examples):
        seqs = examples["sequence"]
        out = tokenizer(
            seqs,
            padding="max_length",
            truncation=True,
            max_length=512
        )

        # 直接映射成整数
        out["task_head_type"] = [task_mapping[t] for t in examples["task_head_type"]]
        out["labels"] = examples["label"]
        return out

    train_dataset = train_dataset.map(tokenize_fn, batched=True)

    print(train_dataset["task_head_type"][:20])
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer)

    # 6) 初始化自定义 T5 多头模型
    config = T5Config.from_pretrained(MODEL_NAME)
    model = T5MultiHeadModel.from_pretrained(
        MODEL_NAME,
        config=config,
        num_classes_binary=2,
        num_classes_10=10
    )

    # 7) 自定义 Trainer
    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model(**inputs)
            loss = outputs["loss"] if has_labels else None
            if prediction_loss_only:
                return (loss, None, None)

            # outputs: {'loss','logits_binary','logits_tenclass','logits_regression'}
            logits_dict = {
                "binary": outputs["logits_binary"],
                "tenclass": outputs["logits_tenclass"],
                "regression": outputs["logits_regression"]
            }
            labels = inputs["labels"] if has_labels else None
            task_types = inputs["task_head_type"]

            return (loss, (logits_dict, task_types), labels)

    def compute_metrics(eval_pred):
        (logits_dict, task_types) = eval_pred[0]
        labels = eval_pred[1]

        import numpy as np
        binary_preds = []
        binary_labels = []
        ten_preds = []
        ten_labels = []
        reg_preds = []
        reg_labels = []

        logits_bin = logits_dict["binary"]
        logits_10 = logits_dict["tenclass"]
        logits_reg = logits_dict["regression"]

        # 转到 CPU
        logits_bin = logits_bin.detach().cpu().numpy() if logits_bin is not None else None
        logits_10 = logits_10.detach().cpu().numpy() if logits_10 is not None else None
        logits_reg = logits_reg.detach().cpu().numpy() if logits_reg is not None else None

        if torch.is_tensor(labels):
            lbls = labels.detach().cpu().numpy()
        else:
            lbls = np.array(labels)

        # 如果 task_types 也是张量，需要映射成字符串；假设目前是 ["binary","tenclass","regression",...]
        # 这里直接当成 list 处理
        n = len(task_types)
        idx_bin = 0
        idx_10 = 0
        idx_reg = 0

        for i in range(n):
            ttype = task_types[i]
            if ttype == "binary":
                pred_probs = logits_bin[idx_bin]  # [2]
                pred_class = np.argmax(pred_probs)
                true_label = lbls[i]
                binary_preds.append(pred_class)
                binary_labels.append(true_label)
                idx_bin += 1
            elif ttype == "tenclass":
                pred_probs = logits_10[idx_10]  # [10]
                pred_class = np.argmax(pred_probs)
                true_label = lbls[i]
                ten_preds.append(pred_class)
                ten_labels.append(true_label)
                idx_10 += 1
            else:  # "regression"
                pred_val = logits_reg[idx_reg]  # scalar
                true_val = lbls[i]
                reg_preds.append(pred_val)
                reg_labels.append(true_val)
                idx_reg += 1

        metrics = {}
        # Binary 准确率
        if len(binary_preds) > 0:
            bin_acc = (np.array(binary_preds) == np.array(binary_labels)).mean()
            metrics["binary_acc"] = bin_acc
        # Ten-class 准确率
        if len(ten_preds) > 0:
            ten_acc = (np.array(ten_preds) == np.array(ten_labels)).mean()
            metrics["tenclass_acc"] = ten_acc
        # 回归 MSE
        if len(reg_preds) > 0:
            reg_mse = ((np.array(reg_preds) - np.array(reg_labels)) ** 2).mean()
            metrics["regression_mse"] = reg_mse

        return metrics

    training_args = TrainingArguments(
        output_dir="/cluster/scratch/jiaxie/models/DeepProtT5",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=1,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        fp16=True
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 开始训练
    trainer.train()

    # 保存最终模型
    trainer.save_model("/cluster/scratch/jiaxie/models/DeepProtT5")
    tokenizer.save_pretrained("/cluster/scratch/jiaxie/models/DeepProtT5")
    print("Done. Model + tokenizer saved.")


if __name__ == "__main__":
    main()
