from src.flowmia import FlowMIA
config_netshare = {
    'member_path': 'datasets/real/celldata/train.csv', # path dos membros
    'non_member_path': 'datasets/reference/telecom_italia_internet.csv', # path dos não-membros
    'synth_path': 'datasets/synthetic/rgan/celldata.csv', # path dos sintéticos
    'test_path': 'datasets/real/celldata/test.csv', # path do teste
    'categorical_cols': [], # colunas categóricas
    'numerical_cols': ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
                       'c8', 'c9'], #colunas numéricas
    'ip_cols': [], # colunas de ip
    'label_col': None, # nome da coluna do rótulo 
    'batch_size': 100, # número de amostrar por lote
    'num_epochs': 100, # número de épocas
    'fcheckpoint': 100, # frequência para salvar o checkpoint
    'save_path': 'resultados_teste/rgan/celldata',    # pasta para salvar resultados
    'use_wgan': True, # se deve usar WGAN ou GAN tradicional
    'test_size': 1000
}

flowmia_netshare = FlowMIA(config=config_netshare)
scores = flowmia_netshare.flowmiagan(test_size = config_netshare['test_size'])