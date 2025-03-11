# CWGrasp
This repository is the official implementation of our __3DV 2025__ paper:

**3D Whole-body Grasp Synthesis with Directional Controllability**
(https://gpaschalidis.github.io/cwgrasp/).

<a href="">
<img src="img/cwgrasp_optimization.gif" alt="Logo" width="100%">
</a>


CWGrasp is a framework for 3D Whole-Body grasping generation either with the right or the
left hand. Given an object upon a receptacle, it first calcualte all the possible reaching 
directions using a probabilistic 3D vector field, called [ReachingField](https://github.com/gpaschalidis/ReachingField). 
It samples a reaching direction from ReachingField and conditioned to that it generates a hand grasp and a reaching body
that satisfy it, using [CGrasp](https://github.com/gpaschalidis/CGrasp) and [CReach](https://github.com/gpaschalidis/CReach) 
respectively. To make the hand from CGrasp and the body from CReach fully compatible we perform optimization 
in the final stage of CWgrasp. 

## Installation & Dependencies
Clone the repository using:

```bash
git clone git@github.com:gpaschalidis/CWGrasp.git
cd CWGrasp
```
Run the following commands:
```bash
conda create -n cwgrasp python=3.9 -y
conda activate cwgrasp
conda install pytorch=2.0.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu117.html
```
```bash
pip install -r requirements.txt
 ```
## Installing Dataset
- Download first the GRAB object mesh (`.ply`) files from the [GRAB website](https://grab.is.tue.mpg.de/).
- Download the ReplicaGrasp dataset from [FLEX github page](https://github.com/purvaten/FLEX).
- Move both datasets inside the folder "data", so you have have the following structure:

```bash
    data
     │
     ├── contact_meshes
     │    ├── airplane.ply
     │    └── ...
     │
     └── replicagrasp
          ├── dset_info.npz
          └── receptacles.npz
```

#### Mano models
- Download MANO models following the steps on the [MANO repo](https://github.com/otaheri/GrabNet) and save them in the folder "models".

#### SMPLX models
- Download body models following the steps on the [SMPLX repo](https://github.com/vchoutas/smplx) and put it in the folder "models".

You should have the following structure:

```bash
    models
      └── mano
      └── smplx
```
## Test CWGrasp
To test CWGrasp you need to download the pre-trained CGrasp and CReach models, as well as the RefineNet model from GrabNet.
- Download the pre-trained CGrasp model from [here](https://docs.google.com/forms/d/1tpUAQms4sAHOj87bsKhPAsCffCj74Y0cD-ywi7pxB_E/edit).
- Download the pre-trained CReach model from [here](https://docs.google.com/forms/d/1TBkvFmiLwf_TAnZOlBEmEXck1Tv-MNUcftlOKxKow1Q/edit).
- Download the pre-trained RefineNet model from the [GRAB website](https://grab.is.tue.mpg.de/).
- Move all these models to the "pretrained" folder, so you have the following structure.

```bash
    pretrained
     ├── cgrasp.pt
     │
     ├── creach.pt
     │
     └── refinenet.pt
```
To run the optimization use the following command:

```bash
 python optimization.py --obj_rec_conf camera_receptacle_aabb_WC1_Top1frl_apartment_wall_cabinet_01_all_0 --config_file cfg/loss_config.yaml --gender "female" --grasp_type "left" --num_samples 3 --save_path $SAVE_FOLDER

```
You can specify the number of samples to generate, the gender of the samples and the grasp type (right or left).

## Citation
If you found this work influential or helpful for your research, please cite:
```
@inproceedings{paschalidis2025cwgrasp,
  title     = {{3D} {W}hole-Body Grasp Synthesis with Directional Controllability},
  author    = {Paschalidis, Georgios and Wilschut, Romana and Anti\'{c}, Dimitrije and Taheri, Omid and Tzionas, Dimitrios},
  booktitle = {{International Conference on 3D Vision (3DV)}},
  year      = {2025}
 }
```
