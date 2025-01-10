If you have obtained the list of data types required by the pyg package, use

python train.py
to obtain the cross-validation results of the GNN-NPI model.

If you want to apply this model to new protein or peptide data, please follow the steps below.

First, submit your protein or peptide sequence to the URL provided in this paper "Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S, Steinegger M. ColabFold: Making Protein Folding Access to All. Nature Methods, 2022" reconstructed from the code provided. We strongly recommend that you run this code on Google Collaboration or other high-performance servers.
Second, you need to download all the PDB files you obtained.

Third, convert the obtained PDB file into the data format data in the pyg package and save it as a data list. At this time, you need to set a threshold to establish edges for amino acids closer than the threshold. This article recommends setting the threshold to 4.5.

Assume that you have obtained the list of data types in the pyg package. Follow the steps below to add all the sequence and structural features used or removed in this paper, or you can modify the feature extraction code to add your own features.
You can then use GNN-NPI.py to iterate over all the selected features and finally get the best combination of features and the corresponding evaluation metrics.
