# ============================================================
# Fine-tuning BERT (Legal-BERT-PTBR) com Agregação de Chunks
# Classificação de temas usando Média de Embeddings + Classe "Não se aplica"
# ============================================================

import re
import pandas as pd
import numpy as np
import polars as pl
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# ============================================================
# 1. Carregar dataframe e Injetar "Não se aplica"
# ============================================================

COLUNA_TEXTO = "texto_chunk"
COLUNA_LABEL = "label"

dataf = pl.read_parquet("dataset_chunks.parquet")
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

df = df.reset_index(drop=True)
df["id_peticao"] = (df["chunk_index"] == 0).cumsum()

# Mantemos todos os chunks para calcular a média posteriormente
df = df.dropna(subset=[COLUNA_TEXTO]).copy()
df[COLUNA_TEXTO] = df[COLUNA_TEXTO].map(limpar_inteiro_teor)

df = df.rename(columns={
    COLUNA_TEXTO: "text",
    COLUNA_LABEL: "label_original"
})

# --- INCLUSÃO DA CLASSE "NÃO SE APLICA" ---
df["label_original"] = df["label_original"].fillna("Não se aplica")

# Agrupa temas raros (menos de 10 ocorrências por petição única) na classe escape
contagem_classes = df.groupby("id_peticao")["label_original"].first().value_counts()
classes_raras = contagem_classes[contagem_classes < 10].index

df["label_original"] = df["label_original"].apply(
    lambda x: "Não se aplica" if x in classes_raras or x == "" else x
)

df["text"] = df["text"].astype(str)

# ============================================================
# 2. Codificar labels
# ============================================================

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label_original"])
num_labels = df["label"].nunique()

id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in id2label.items()}

print("Número de classes (com 'Não se aplica'):", num_labels)

# ============================================================
# 3. Separar treino e teste (Garante agrupamento por petição)
# ============================================================

ids_peticoes = df["id_peticao"].unique()
train_ids, test_ids = train_test_split(ids_peticoes, test_size=0.2, random_state=42)

train_df = df[df["id_peticao"].isin(train_ids)].copy()
test_df = df[df["id_peticao"].isin(test_ids)].copy()

train_dataset = Dataset.from_pandas(train_df[["text", "label", "id_peticao"]], preserve_index=False)
test_dataset = Dataset.from_pandas(test_df[["text", "label", "id_peticao"]], preserve_index=False)

# ============================================================
# 4. Tokenização 
# ============================================================

MODEL_NAME = "dominguesm/legal-bert-base-cased-ptbr"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Deixamos apenas 'input_ids', 'attention_mask', 'token_type_ids' e 'labels'
train_dataset = train_dataset.remove_columns(["text", "id_peticao"])
test_dataset = test_dataset.remove_columns(["text", "id_peticao"])

# Renomeia label para labels, que é o nome esperado pelo Trainer/modelo
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================
# 5. Modelo Customizado: Média de Embeddings de Chunks
# ============================================================

class BertMeanEmbeddingClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask, labels=None, id_peticao=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling dos tokens do chunk atual
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        chunk_embeddings = sum_embeddings / sum_mask # Vetor de 768 para cada chunk
        
        # Agregação por média das petições contidas no lote
        # Para manter compatibilidade com o Trainer do HuggingFace, se id_peticao não for consolidador dinâmico,
        # passamos direto para o classificador linear por chunk (mapeando a label do doc em todos os seus pedaços)
        logits = self.classifier(chunk_embeddings)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

model = BertMeanEmbeddingClassifier(MODEL_NAME, num_labels=num_labels)

# ============================================================
# 6. Métricas
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1
    }

# ============================================================
# 7. Argumentos de treino e Execução (Corrigido)
# ============================================================

training_args = TrainingArguments(
    output_dir="./bert_mean_embeddings",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    
    # warmup_ratio=0.1, <-- REMOVA ESTA LINHA
    warmup_steps=100,     # <-- ADICIONE ESTA LINHA (100 passos de aquecimento inicial)
    
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# ============================================================
# 10. Avaliar e Salvar
# ============================================================
resultados = trainer.evaluate()
print("Resultados Finais com Média de Embeddings:", resultados)

model.bert.save_pretrained("./modelo_bert_media_temas")
tokenizer.save_pretrained("./modelo_bert_media_temas")

import joblib
joblib.dump(label_encoder, "./modelo_bert_media_temas/label_encoder.pkl")