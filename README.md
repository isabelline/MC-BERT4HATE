# MC-BERT4HATE
Repository for the paper "MC-BERT4HATE: Hate Speech Detection using Multi-channel BERT for Different Languages and Translations"

## Download checkpoints and config files
- [Multilingual-bert](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip)
- [Base-bert(English)](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip)
- [Chinese-bert](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

## Run
```
python run_mcbert.py \
      --do_train=true \
      --bert_config_file=/content/drive/My Drive/bertconfigmulti/bert_config.json \
      --bert_config_file_en=/content/drive/My Drive/bertconfig12/bert_config.json \
      --bert_config_file_ch=/content/drive/My Drive/bertconfigchinese/config/bert_config.json \
      --output_dir=/content/trial_one \
      --vocab_file=/content/drive/My Drive/bertconfigmulti/vocab.txt \
      --vocab_file_en=/content/drive/My Drive/bertconfig12/vocab.txt \
      --vocab_file_ch=/content/drive/My Drive/bertconfigchinese/config/vocab.txt \
      --do_lower_case=false \
      --data_dir=/content/drive/My Drive/hateval/es/ \
      --max_seq_length=50 \
      --init_checkpoint=/content/drive/My Drive/bertconfigmulti/bert_model.ckpt \
      --init_checkpoint_en=/content/drive/My Drive/bertconfig12/bert_model.ckpt \
      --init_checkpoint_ch=/content/drive/My Drive/bertconfigchinese/config/bert_model.ckpt
```

## Reference
```
@inproceedings{inproceedings,
author = {Sohn, Hajung and Lee, Hyunju},
year = {2019},
month = {11},
pages = {551-559},
title = {MC-BERT4HATE: Hate Speech Detection using Multi-channel BERT for Different Languages and Translations},
doi = {10.1109/ICDMW.2019.00084}
}
```
