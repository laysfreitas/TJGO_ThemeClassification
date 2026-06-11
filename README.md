# TJGO_ThemeClassification

Código para classificar temas jurídicos baseados nas seguintes informações: Informação Legislativa, Ementa , Inteiro teor. 
Cada linha terá textos com 512 tokens ( pois usaremos o modelo BERT para fazer o fine tunning ) e cada texto terá seu respectivo label.

Será retirado as palavras comuns nas 3 informações, com o objetivo de diminuir os ruídos ao treinar o modelo.

Será aplicado também a técnica de SupCon loss na fase de treinamento.

Ao final terá uma interface onde uma petição poderá ser inserida e retornará o possível tema e a porcentagem de acerto inferido pelo modelo.
