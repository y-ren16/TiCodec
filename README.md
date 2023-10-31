# TiCodec
PyTorch Implementation of
[Fewer-token Neural Speech Codec with Time-invariant Codes.](https://arxiv.org/abs/2310.00014)
## Demo Pages
https://y-ren16.github.io/TiCodec is the demo page of the paper 'Fewer-token Neural Speech Codec with Time-invariant Codes'.
## Quick Started
### Dependencies
```
pip install -r requirement.txt
```
Install [visqol](visqol) for calculating objective metrics.
### dataset preparation
Gengenerate ``./Lib_resources/*/*.lst`` for training.
```
.
├── Lib_resources
│   └── LibriTTS
│       ├── dev.lst
│       ├── test.lst
│       └── train.lst
```
Put ``*.wav`` in ``./egs/Paper_Data/GT/*/*.wav`` for metrics.
```
.
├── Paper_Data
│   ├── GEN
│   └── GT
```
### Train
```
cd TiCodec/egs/TiCodec-24k-320d
bash start_conv_1g1r_8g3k1s_cos_from_head.sh
bash start_conv_1g2r_8g3k1s_cos_from_head.sh
bash start_conv_1g4r_8g3k1s_cos_from_head.sh
```
### Test
```
test_conv_1g1r_8g3k1s_cos_from_head.sh
test_conv_1g2r_8g3k1s_cos_from_head.sh
test_conv_1g4r_8g3k1s_cos_from_head.sh
```
### Metrics
```
python metrics.py
```
## Acknowledgements
This implementation uses parts of the code from the following Github repos: [AcademiCodec](https://github.com/yangdongchao/AcademiCodec)
## Citations
If you find this code useful in your research, please consider citing:
```
@article{ren2023fewer,
  title={Fewer-token Neural Speech Codec with Time-invariant Codes},
  author={Ren, Yong and Wang, Tao and Yi, Jiangyan and Xu, Le and Tao, Jianhua and Zhang, Chuyuan and Zhou, Junzuo},
  journal={arXiv preprint arXiv:2310.00014},
  year={2023}
}
```
