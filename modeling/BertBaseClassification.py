# ============================================================
# Fine-tuning BERT-base sem SupCon Loss
# Classificação de temas usando Cross Entropy
# ============================================================

import re
import pandas as pd
import numpy as np
import polars as pl
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# ============================================================
# 1. Carregar dataframe
# ============================================================

# Dataframe precisa ter pelo menos:
# - uma coluna de texto
# - uma coluna de label/tema

# Ajuste aqui para os nomes reais das colunas
COLUNA_TEXTO = "texto"
COLUNA_LABEL = "label"

dataf = pl.read_parquet(r"C:\Users\lfmelo\Documents\Github\TJGO_ThemeClassification\data\dataset_base.parquet")
df = dataf.to_pandas()

_INTEIRO_TEOR_PATTERNS = [
    (re.compile(r"(Erro Parser)?>>>>>inicio<<<<<\n?", re.MULTILINE | re.IGNORECASE), ""),
    (re.compile(r"fimid:\d+|#####fim#####id:\d+\n?", re.MULTILINE), ""),
]
_MULTI_SPACE = re.compile(r" {2,}")

def limpar_inteiro_teor(text: str) -> str:
    for pattern, replacement in _INTEIRO_TEOR_PATTERNS:
        text = pattern.sub(replacement, text)
    return _MULTI_SPACE.sub(" ", text).strip()

df = df.dropna(subset=["texto"]).copy()
df["texto"] = df["texto"].map(limpar_inteiro_teor)

df = df.rename(columns={
    COLUNA_TEXTO: "text",
    COLUNA_LABEL: "label_original"
})

# Garante que o texto está como string
df["text"] = df["text"].astype(str)

# ============================================================
# 2. Codificar labels
# ============================================================

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label_original"])

num_labels = df["label"].nunique()

id2label = {
    i: label
    for i, label in enumerate(label_encoder.classes_)
}

label2id = {
    label: i
    for i, label in id2label.items()
}

print("Número de classes:", num_labels)
print("Exemplo de mapeamento:", id2label)

# ============================================================
# 3. Separar treino e teste
# ============================================================

train_df, test_df = train_test_split(
    df[["text", "label"]],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

# O Hugging Face Datasets permite dividir, mapear e transformar datasets
# usando funções como train_test_split e map. 
# Aqui usamos Dataset.from_pandas porque partimos de um DataFrame.
# Referência: documentação oficial de processamento do Datasets. 
# https://huggingface.co/docs/datasets/process

# ============================================================
# 4. Tokenização
# ============================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove a coluna de texto original depois da tokenização
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

# Renomeia label para labels, que é o nome esperado pelo Trainer/modelo
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Padding dinâmico: completa cada batch conforme o maior texto daquele batch,
# em vez de deixar tudo com padding fixo.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================
# 5. Criar modelo BERT para classificação
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# ============================================================
# 6. Métricas
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1
    }

# ============================================================
# 7. Argumentos de treino
# ============================================================

training_args = TrainingArguments(
    output_dir="./bert_base_cross_entropy",
    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,
    weight_decay=0.01,

    logging_dir="./logs",
    logging_steps=50,

    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,

    save_total_limit=2,

    report_to="none"
)

# ============================================================
# 8. Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ============================================================
# 9. Treinar
# ============================================================

trainer.train()

# ============================================================
# 10. Avaliar
# ============================================================

resultados = trainer.evaluate()
print(resultados)

# ============================================================
# 11. Salvar modelo final
# ============================================================

trainer.save_model("./modelo_bert_base_temas")
tokenizer.save_pretrained("./modelo_bert_base_temas")

# Também salva o encoder das labels
import joblib
joblib.dump(label_encoder, "./modelo_bert_base_temas/label_encoder.pkl")