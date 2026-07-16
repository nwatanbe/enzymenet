# EnzymeNet 
Source code for enzyme function prediction and candidate enzyme prioritization. 

1.  System requirements
2.  Installation guide
3.  Demo 
4.  Instructions for use
5.  License
6.  Related publications

# 1.  System requirements
Any Linux operating system with Anaconda is recommended for use.  
The software was tested on CentOS 7.5.1804 (Core) with Anaconda3-2019.10.  
python version > 3.7.4
  
No non-standard hardware is required.  
  
# 2.  Installation guide
EnzymeNet is downloaded from the following link:  
https://github.com/nwatanbe/enzymenet 
 
Conda environment is built using the following commands:  
conda create –n enzymenet python=3.7.4  
conda activate enzymenet  

Other packages areinstalled as described in "/Downloaded_directory_path/enzymenet/env/requirements.txt"  
pip install -r /Downloaded_directory_path/enzymenet/env/requirements.txt  
Downloaded_directory_path should be given an arbitrary directory path containing EnzymeNet.   

The software typically takes several minutes to install.  

EnzymeNet model (model.tar.gz) can be downloaded from the following link:  
https://drive.google.com/drive/folders/1mk_SFD7fRDtZTT_mmKRq1Kwa1wg9lnCV 
or
https://drive.google.com/drive/folders/1DkBpl_0GWlmILFJXksyOa974HGsx2eic
  
The "model.tar.gz" is decompressed using the following command:  
tar -zxvf model.tar.gz  
  
The decompressed directory is added to /Downloaded_directory_path/enzymenet/  
  
# 3.  Demo
The directory including in EnzymeNet script is changed into:  
cd /Downloaded_directory_path/enzymenet/script/ 
Downloaded_directory_path should be given an arbitrary directory path containing EnzymeNet.  
 
## Enzyme function prediction
Sample test file: /Downloaded_directory_path/enzymenet/data/select_samples_for_ec_predict.fasta  
  
EnzymeNet is run for prediction of test samples by the following command:  
./test_ec.sh  

The following results are output into"/Downloaded_directory_path/enzymenet/result/ec_number/EC_predict_final_result.tsv":  
●	Prediction results of Enzyme Commission (EC) number 1st digit and score  
●	Prediction results of complete EC number and score  
 
The demo run time is several minutes.  
Up to 4,000 sequences are predicted in about 10 minutes at one time. 

## Candidate enzyme prioritization
Sample search file: /Downloaded_directory_path/enzymenet/data/select_samples_for_vec_search.fasta  
Sample reference file: /Downloaded_directory_path/enzymenet/data/select_samples_for_vec_ref.fasta  
  
EnzymeNet is run for selection of enzyme candidates, by the following command:  
./test_vv.sh  

The following results are output into"/Downloaded_directory_path/enzymenet/result/vectorize_visualize/":  
●	Enzyme candidate ranking and similarity score results  
●	Figure showing enzyme candidates and reference data  
  
# 4.  Instructions for use
Run time depends on the dataset size.  
## Enzyme function prediction
When you want to change input file, you should change it into new file name in "/Downloaded_directory_path/enzymenet/script/test_ec.sh".  
You have to include FASTA file in "/Downloaded_directory_path/enzymenet/data/" 

## Candidate enzyme prioritization
When you want to change 2 input files, you should change them into new file names in "/Downloaded_directory_path/enzymenet/script/test_vv.sh".  
You have to include 2 FASTA files in "/Downloaded_directory_path/enzymenet/data/" 
  
# 5.  License
This software is released under the MIT License, according to LICENSE.txt.  
 
# 6. Related publications
1. Naoki Watanabe., Masaki Yamamoto., Masahiro Murata., Yuki Kuriya., and Michihiro Araki.  
   EnzymeNet: Residual Neural Networks model for Enzyme Commission number prediction.  
   Bioinformatics Advances, 2025.  
   DOI: 10.1093/bioadv/vbad173   
3. Naoki Watanabe., Shuhei Noda., et al.  
   Enzyme Analyzer: A web server for efficient search of candidate enzyme sequences in bioproduction of functional compounds.  
   Manuscript in preparation.  
