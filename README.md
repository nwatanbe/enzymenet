# EnzymeNet: Enzyme Comission (EC) number prediction
Naoki Watanabe, Masaki Yamamoto, Masahiro Murata, Yuki Kuriya, and Michihiro Araki


1.  System requirements
2.  Installation guide
3.  Demo 
4.  Instructions for use
5.  License

# 1.  System requirements
Any Linux operating system with Anaconda is recommended for use.  
The software was tested on CentOS 7.5.1804 (Core) with Anaconda3-2019.10.  
python version > 3.7.4
  
No non-standard hardware is required.  
  
# 2.  Installation guide
Anaconda can be downloaded from the following link:  
https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh  

Anaconda can be installed using the following command:  
./Downloaded_directory_path/Anaconda3-2019.10-Linux-x86_64.sh  

"Downloaded_directory_path" should be given an arbitrary directory path containing Anaconda.  

EnzymeNet is downloaded from the following link:  
https://github.com/nwatanbe/enzymenet 
 
Conda environment is built using the following commands:  
conda create –n enzyme python=3.7.4  
conda activate enzyme  

Other packages areinstalled as described in "/Downloaded_directory_path/enzymenet/env/requirements.txt"  
pip install -r /Downloaded_directory_path/enzymenet/env/requirements.txt  
Downloaded_directory_path should be given an arbitrary directory path containing EnzymeNet.   

The software typically takes several minutes to install.  

EnzymeNet model (model.tar.gz) can be downloaded from the following link:  
https://drive.google.com/drive/folders/1mk_SFD7fRDtZTT_mmKRq1Kwa1wg9lnCV  
  
The "model.tar.gz" is decompressed using the following command:  
tar -zxvf model.tar.gz  
  
The decompressed directory is added to /Downloaded_directory_path/enzymenet/  
  
# 3.  Demo
The directory including in EnzymeNet script is changed into:  
cd /Downloaded_directory_path/enzymenet/script/ 
Downloaded_directory_path should be given an arbitrary directory path containing EnzymeNet.  
 
EnzymeNet is run for prediction of test samples (/Downloaded_directory_path/enzymenet/data/select_samples_for_ec_predict.fasta), by the following command:  
./test_ec.sh 

The following results are output into"/Downloaded_directory_path/enzymenet/result/":  
●	Prediction results of EC number 1~s~t digit and score  
●	Prediction results of complete EC number and score  
 
The demo run time is several minutes.  
Up to 4,000 sequences are predicted in about 10 minutes at one time. 
 
# 4.  Instructions for use
When you want to change input file, you should change "select_samples_for_ec_predict.fasta" into "new file name" in "/Downloaded_directory_path/enzymenet/script/test_ec.sh".  
You have to include FASTA file in "/Downloaded_directory_path/enzymenet/data/" 
Run time depends on the dataset size. 

# 5.  License
This software is released under the MIT License, according to LICENSE.txt. 
