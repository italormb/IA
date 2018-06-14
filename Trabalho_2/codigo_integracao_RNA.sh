#!/bin/bash

#execute no terminal
#chmod +x codigo_integracao_RNA.sh
#./codigo_integracao_RNA.sh



#Baixando o pacote python 3.5  
sudo apt-get install python3.5
#importando o pacote para plotar
sudo apt-get install python-matplotlib
#biblioteca usada na matriz de confusao
sudo pip install -U scikit-learn
#baixando o keras que faz o processamento de imagem
sudo pip install keras
#entrando no ambiente python3
source activate py36
python PNEUMONIA_RNA_TRAIN_V2.py 
python PNEUMONIA_RNA_MATRIZ_CONFUSAO_V2.py
python PNEUMONIA_RNA_IMAGEM_V2.py
echo