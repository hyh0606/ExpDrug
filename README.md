# ExpDrug

## Requirements

- PyTorch 1.13.0 and CUDA 12.2.
- The learning rate of 5e−4 for 50 epochs.

## Folder Specification

### Dataset Preparation

For the MIMIC-III dataset, the following files are required: 
(Here, we do not share the MIMIC-III dataset due to reasons of personal privacy, maintaining research standards, and legal considerations.)  Go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset .

- MIMIC-III
  - PRESCRIPTIONS.csv
  - DIAGNOSES_ICD.csv 
  - PROCEDURES_ICD.csv 
  - D_ICD_DIAGNOSES.csv
  - D_ICD_PROCEDURES.csv

- data/: we use the same data set and processing methods as safedrug2021 https://github.com/ycq091044/SafeDrug

### Step1 :Processing file

#### data/data_new/

- **ndc2atc_level4.csv**: this is a NDC-RXCUI-ATC4 mapping file, and we only need the RXCUI to ATC4 mapping. This file is obtained from https://github.com/ycq091044/SafeDrug.
- **drug-atc.csv**: this is a CID-ATC file, which gives the mapping from CID code to detailed ATC code . This file is obtained fromhttps://github.com/ycq091044/SafeDrug.
- **ndc2rxnorm_mapping.txt**: NDC to RXCUI mapping file. This file is obtained from https://github.com/ycq091044/SafeDrug.
- **drug-DDI.csv**: this a large file, containing the drug DDI information, coded by CID. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing

#### data/data_new/

- run processing.py（processing  the  data  to  get  voc_final.pkl, records_final_131.pkl,  ddi_A_final.pkl  and  ehr_adj_final.pkl ）

``````
python processing.py
``````

- run matrix.py （processing  the  data  to  get  diag_new.pkl,  pro_new.pkl  and  med131_new.pkl ）

### Step 2: run the code

#### src/

- run ExpDrug_main.py

```python  ExpDrug_main.py
python ExpDrug_main.py
```