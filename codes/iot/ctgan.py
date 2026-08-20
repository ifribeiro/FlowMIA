import sys
import os

# Adiciona a pasta raiz do projeto (onde está a pasta 'src') ao caminho de busca
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.flowmia import FlowMIA
config_netshare = {
    'member_path': 'datasets/real/iot_treino.csv', # path dos membros
    'non_member_path': 'datasets/reference/ton.csv', # path dos não-membros
    'synth_path': 'datasets/synthetic/ctgan/iot/synth_BS_200.csv', # path dos sintéticos
    'test_path': 'datasets/real/iot_test.csv', # path do teste
    'categorical_cols': ['proto'], # colunas categóricas
    'numerical_cols': ['srcport', 'dstport', 'td', 'pkt', 'byt'], 
    'ip_cols': ['srcip', 'dstip'], # colunas de ip
    'label_col': 'label', # nome da coluna do rótulo 
    'batch_size': 100, # número de amostrar por lote
    'num_epochs': 10, # número de épocas
    'fcheckpoint': 100, # frequência para salvar o checkpoint
    'save_path': 'resultados_teste/ctgan/iot23',    # pasta para salvar resultados
    'use_wgan': True, # se deve usar WGAN ou GAN tradicional
    'test_size': 1000
}

flowmia_netshare = FlowMIA(config=config_netshare)
scores = flowmia_netshare.flowmiagan(test_size = config_netshare['test_size'])