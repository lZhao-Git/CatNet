# CatNet
Our research presents the CatNet model, a novel advance that synergistically integrates chemical and protein information to predict interactions between chemicals and NRs. What sets CatNet apart is its scalability and ability to maintain accuracy even on uncharted NR territories.
<div align=center><image width="535" height="200" src="https://github.com/lZhao-Git/CatNet/blob/master/pics/framework.png"/></div>

## Requirements
- PyTorch = 1.12.1
-  scikit_learn = 0.24.0
- rdkit = 2022.9.2
- numpy = 1.23.4
- pandas = 1.5.1
## Usage
1. Prepare dataset
`py mol_featurizer.py`
2. Train model
`py main.py`
3. Graphical user interface (GUI)
`py CatNet.py`
<div align=center><image width="550" height="400" src="https://github.com/lZhao-Git/CatNet/blob/master/pics/GUI.png"/></div>
To use it, simply input the SMILES string of the chemical and the amino acid sequence of the desired protein. Once entered, just click on the "Submit" button, and the software will display the predicted outcome in the "Result of prediction" section. For those handling bulk data, CatNet provides a feature where users can conveniently upload a txt file containing multiple SMILES strings and corresponding protein sequences. Once predictions are complete, results can be effortlessly downloaded in CSV format.
  
## Citation
Lu Zhao, Qiao Xue*, Huazhou Zhang, Yuxing Hao, Hang Yi, Xian Liu, Wenxiao Pan, Jianjie Fu, Aiqian Zhang*. CatNet: Sequence-based deep learning with cross-attention mechanism for identifying endocrine-disrupting chemicals [J]. Journal of Hazardous Materials, 2024, 465: 133055
