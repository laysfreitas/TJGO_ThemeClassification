import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CAMINHO_MODELO = r"C:\Users\lfmelo\Documents\Github\TJGO_ThemeClassification\modeling\modelo_bert_base_temas"

tokenizer = AutoTokenizer.from_pretrained(CAMINHO_MODELO)
model = AutoModelForSequenceClassification.from_pretrained(CAMINHO_MODELO)
label_encoder = joblib.load(f"{CAMINHO_MODELO}/label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def prever_tema(texto):
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()

    tema_predito = label_encoder.inverse_transform([pred_id])[0]
    confianca = probs[0][pred_id].item()

    return {
        "tema_predito": tema_predito,
        "classe_id": pred_id,
        "confianca": confianca
    }

# Exemplo
texto_teste = "Insira aqui o texto da petição ou documento jurídico."
resultado = prever_tema(texto_teste)

print(resultado)