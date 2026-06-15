import torch
import joblib

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# Configurações
# ============================================================

CAMINHO_MODELO = "./modelo_bert_base_temas"

app = Flask(__name__)


# ============================================================
# Carregar modelo uma única vez
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(CAMINHO_MODELO)
model = AutoModelForSequenceClassification.from_pretrained(CAMINHO_MODELO)
label_encoder = joblib.load(f"{CAMINHO_MODELO}/label_encoder.pkl")

model.to(device)
model.eval()


# ============================================================
# Função de inferência
# ============================================================

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
        confianca = probs[0][pred_id].item()

    tema_predito = label_encoder.inverse_transform([pred_id])[0]

    return {
        "tema_predito": str(tema_predito),
        "classe_id": int(pred_id),
        "probabilidade": round(confianca * 100, 2)
    }


# ============================================================
# Rotas
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prever", methods=["POST"])
def prever():
    dados = request.get_json()

    texto = dados.get("texto", "").strip()

    if not texto:
        return jsonify({
            "erro": "Digite o texto da petição antes de prever."
        }), 400

    resultado = prever_tema(texto)

    return jsonify(resultado)


# ============================================================
# Rodar aplicação
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)