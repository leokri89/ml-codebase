# DVC

## Introdução

Controle de Versão para Dados: DVC é uma ferramenta de código aberto que permite versionar grandes conjuntos de dados e modelos de machine learning, similar ao Git, mas focado em dados.

Gerenciamento de Experimentos: Facilita o rastreamento de experimentos de machine learning, permitindo comparar diferentes versões de modelos e dados.

Integração com Armazenamento Remoto: Suporta diversos tipos de armazenamento remoto, como Amazon S3, Google Drive e Azure Blob Storage, para facilitar o compartilhamento e a colaboração.

Pipelines Reproduzíveis: Permite a criação de pipelines de dados reproduzíveis, garantindo que os experimentos possam ser replicados e auditados facilmente.

## Comandos

### Inicializar o repositório:
```sh
dvc init
git status
git commit -m "Inicializa DVC"
```

### Adicionar arquivos para versionamento no DVC:
```sh
dvc add data/data.csv
git add data/data.xml.dvc data/.gitignore
git commit -m "Adiciona datasets"
```

### Adicionar servidor remoto:
```bash
# Remoto local para testes
dvc remote add -d my_remote /tmp/dvcstore

# Remoto no S3
dvc remote add -d s3_storage s3://mybucket/dvcstore

# Remoto no Google Drive
dvc remote add --default gdrive_storage gdrive://12FEUHHm3BnYb7iGAQ8Hgu0bUcaMfHoc4/dvcstore
dvc remote modify gdrive_storage gdrive_acknowledge_abuse true
```

### Envia dados para o servidor remoto:
```bash
dvc push
```

### Recuperando dados do servidor remoto:
```bash
dvc pull
```

### Atualizar dados:
```bash
dvc add data/data.xml
dvc push
git commit data/data.xml.dvc -m "Dataset updates"
```

### Trocando entre versões:
```bash
git checkout branch
dvc checkout
```

### Rollback de versão:
```bash
git checkout HEAD~1 data/data.xml.dvc
dvc checkout
git commit data/data.xml.dvc -m "Revert dataset updates"
```
