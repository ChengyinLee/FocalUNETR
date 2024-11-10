# FocalUNETR
A Focal Transformer for Boundary-aware Prostate Segmentation using CT Images (MICCAI20223, [paper link](https://arxiv.org/abs/2210.03189))
![Alt Text](focalunetr.png)
## Data Preprocessing
- You can refer to [this link](https://github.com/llmir/MultitaskOCTA) for generating boundaries from the ground truth mask of the organ you would like to segment.
- Please follow [this reference](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) regarding your original datasets for conversion, such as operations for resampling, cropping, and padding.
## For Training and Testing
- Based on the settings in the options folder containing `yml` files, training and testing can be performed with or without using the boundary-aware regression auxiliary task. 
- For additional baseline models, please refer to [this link](https://github.com/yhygao/CBIM-Medical-Image-Segmentation).
## Contact
- Chengyin Li, cyli@wayne.edu
