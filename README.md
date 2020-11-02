# Remember to put these two directories (./data and ./res_data/results) into program (same path of this readme)

# Scripts:
- Would be executed:
  - config_training.py: set filepaths 
  - make_dataset.py: generate 5-fold datasplits
  - prepare.py: LUNA16 dataset preprocessing 
  - train_detector.sh: excecuting main_detector_recon.py
  - main_detector_recon.py: model training and inference
  - GenerateCSV.py: Generate result.csv for computing CPM
  - noduleCADEvaluaionLUNA16.py: Compute CPM of .csv
  - FROC_CPM.ipynb: Plot FROC curve
  - ./net/sCaps.py: sCaps ; ./net/dCaps.py: dCaps ; ./net/res18.py: ResNet-18 ; ./net/densenet.py: densenet
- Others
  - data_detector.py: generate data loader during training and testing (super difficult to undetstand)
  - preprocess.py: some preproceesing-related codes (may be unused in LUNA16 data)
  - layers.py
  - loss.py: for sCaps.py and dCaps.py 
  - loss_2.py: for res18.py and densenet.py
  - split_combine.py (At Testing stage)
  - utils.py
  - learning_rates.py: Snapshot Ensembles (Edited by Dr.Chen)

# Requirements:
- Python 3.6
- torch 0.4.1
- torchvision 0.2.0

# Files:
- LUNA.json: Every Case ID stored in .json for make_dataset.py
- test_0222_1 ~ test_0222_5: Five dir containing train/val/test.json (Case ID) respectively
- ./data/LUNA16/allset: all .raw and .mhd of LUNA16 data
- ./data/LUNA16/seg-lungs-LUNA16: all .zraw and .mhd of LUNA16 mask 

# How to Do step by step:
- Randomly generate 5-fold datasplit 
  - python make_dataset.py -> Can change save_dir in main()
  ```
    Output: Five dir containing train/val/test.json (Patient ID) respectively
  ```
- Preprocessing for LUNA16
  - python prepare.py
  - output file path: config_training -> config[preprocess_result_path]
  ```
    Output: id_clean.npy & id_label.npy (for training) ; id_extendbox.npy & id_mask.npy & id_origin.npy & id_spacing.npy (for vox2world) 
  ```
- Ready for Training and Testing
  - Modify config['crop_size'] in sCaps.py (or dCaps.py, res18.py ...) at training stage
  - Modify margin & sidelen in main_detector_recon.py (VOI Size for test = margin*2 + sidelen)

- Start training and testing
  ```
    bash train_detector.sh
  ```
  - training
  ```
    python main_detector_recon.py --model sCaps -b [batch_size] --epochs [num_epochs] --save-dir [save_dir_path] --save-freq [save_freq_ckpt] --gpu '0' --n_test [number of gpu for test] --lr [lr_rate] --cross [1-5 set which cross_data be used] #--resume [resume ckpt]
  ```
  - testing
  ```
    python main_detector_recon.py --model sCaps --resume [resume_ckpt] --save-dir [] --test 1 --gpu '0' --n_test [] --cross []
  ```
  ```
    output id_lbb.npy (label), id_pbb.npy (predicted bboxes)
  ```

- Compute CPM
  - Generate result.csv
  ```
    python GenerateCSV.py
  ```
  ```
    output: sCaps_80_all.csv
  ```
  - Then compute CPM and save related .png and .npy 
  ```
    python noduleCADEvaluaionLUNA16.py (Remember to modify the filepath in noduleCADEvaluation.py)
  ```
  ```
    output: print(csv_name, CPM, seven_sensitivities@predefined_fps) and save ./CPM_Results_sCaps_80/_.npy & _.png
  ```

- Plot FROC Curve
  ```
    Execute FROC_CPM.ipynb
  ```
  ```
    output: FROC_sCaps.png
  ```

- How to transform voxel coord of pbb into world voxel of 3D CT ?
  - You can refer to GenerateCSV.py: How to transform pbb -> pos
