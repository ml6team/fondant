# Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network
## Introduction
```
@inproceedings{wang2019efficient,
  title={Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network},
  author={Wang, Wenhai and Xie, Enze and Song, Xiaoge and Zang, Yuhang and Wang, Wenjia and Lu, Tong and Yu, Gang and Shen, Chunhua},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8440--8449},
  year={2019}
}
```

Note that, the original PAN is based on Python 2.7 and Pytorch 0.4.1.
When migrating it to Python 3.6 and Pytorch 1.1.0, we make the following two changes to the default settings.
- Using Adam optimizer;
- PolyLR is also used in the pre-training phase.

## Results and Models
- Total-Text

| Method | Backbone | Finetune | Precision (%) | Recall (%) | F-measure (%) | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PAN | ResNet18 | N | 87.9 | 79.6 | 83.5 | [config](pan_r18_tt.py) | [model](https://drive.google.com/file/d/1YH4OeftQeFNKKafR1oxRyyT_2MRlRN_n/view?usp=sharing) |
| PAN (paper) | ResNet18 | N | 88.0 | 79.4 | 83.5 | - | - |
| PAN | ResNet18 | Y | 88.5 | 81.7 | 85.0 | [config](pan_r18_tt_finetune.py) | [model](https://drive.google.com/file/d/1bWBTIfmlMd5zUy0b5YL4g8erDgSuLfNN/view?usp=sharing) |
| PAN (paper) | ResNet18 | Y | 89.3 | 81.0 | 85.0 | - | - |

- CTW1500

| Method | Backbone | Finetune | Precision (%) | Recall (%) | F-measure (%) | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PAN | ResNet18 | N | 85.1 | 79.1 | 82.0 | [config](pan_r18_ctw.py) | [model](https://drive.google.com/file/d/1qq7-MI1bOCykKj95uqjqkITa-nmXjinT/view?usp=sharing) |
| PAN (paper) | ResNet18 | N | 84.6 | 77.7 | 81.0 | - | - |
| PAN | ResNet18 | Y | 86.0 | 81.0 | 83.4 | [config](pan_r18_ctw_finetune.py) | [model](https://drive.google.com/file/d/1UY0K2JPsUmqmaJ68k2Q6KwByhogF1Usv/view?usp=sharing) |
| PAN (paper) | ResNet18 | Y | 86.4 | 81.2 | 83.7 | - | - |

- ICDAR 2015

| Method | Backbone | Finetune | Precision (%) | Recall (%) | F-measure (%) | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PAN | ResNet18 | N | 84.4 | 77.5 | 80.8 | [config](pan_r18_ic15.py) | [model](https://drive.google.com/file/d/1dHiXRyreSAG0vqbLyJ0PJfnj56l_P6WZ/view?usp=sharing) |
| PAN (paper) | ResNet18 | N | 82.9 | 77.8 | 80.3 | - | - |
| PAN | ResNet18 | Y | 86.6 | 79.7 | 83.0 | [config](pan_r18_ic15_finetune.py) | [model](https://drive.google.com/file/d/13m7hPZ8mhffaQwch_U6XPOvIG2ouNKHD/view?usp=sharing) |
| PAN (paper) | ResNet18 | Y | 84.0 | 81.9 | 82.9 | - | - |

- MSRA-TD500

| Method | Backbone | Finetune | Precision (%) | Recall (%) | F-measure (%) | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PAN | ResNet18 | N | 82.0 | 79.4 | 80.7 | [config](pan_r18_msra.py) | [model](https://drive.google.com/file/d/1dUf9YH8tPuzijH5-7Ul6Vl6jTq5ziObJ/view?usp=sharing) |
| PAN (paper) | ResNet18 | N | 80.7 | 77.3 | 78.9 | - | - |
| PAN | ResNet18 | Y | 85.7 | 83.4 | 84.5 | [config](pan_r18_msra_finetune.py) | [model](https://drive.google.com/file/d/1csNqq__MqAwug5XRC3L40fh5urLaL0IZ/view?usp=sharing) |
| PAN (paper) | ResNet18 | Y | 84.4 | 83.8 | 84.1 | - | - |

- SynthText

| Method | Backbone |           Config           |                           Download                           |
| :----: | :------: | :------------------------: | :----------------------------------------------------------: |
|  PAN   | ResNet18 | [config](pan_r18_synth.py) | [model](https://drive.google.com/file/d/1TpNg7CQKH1Lh4k9mmDDIzds3WBVDIbu8/view?usp=sharing) |

