# FAST: Faster Arbitrary-Shaped Text Detector with Minimalist Kernel Representation

## Results and Models

**IC17-MLT Pretrained FAST Models**

| Model | Backbone | Pretrain | Resolution | #Params | Config| Download |
| :---: |  :---: | :---: | :---: | :---: |  :---: | :---: |
| FAST-T | TextNet-T | [ImageNet-1K](https://github.com/czczup/FAST/releases/download/release/fast_tiny_in1k_epoch_299.pth)  | 640x640 | 8.5M  | [config](ic17mlt/fast_tiny_ic17mlt_640.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_ic17mlt_640.pth) \| [log](../../logs/ic17mlt/fast_tiny_ic17mlt_640.txt)   |
| FAST-S | TextNet-S | [ImageNet-1K](https://github.com/czczup/FAST/releases/download/release/fast_small_in1k_epoch_299.pth) | 640x640 | 9.7M  | [config](ic17mlt/fast_small_ic17mlt_640.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_ic17mlt_640.pth) \| [log](../../logs/ic17mlt/fast_small_ic17mlt_640.txt) |
| FAST-B | TextNet-B | [ImageNet-1K](https://github.com/czczup/FAST/releases/download/release/fast_base_in1k_epoch_299.pth)  | 640x640 | 10.6M | [config](ic17mlt/fast_base_ic17mlt_640.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ic17mlt_640.pth) \| [log](../../logs/ic17mlt/fast_base_ic17mlt_640.txt)   |

**Results on Total-Text**

| Method | Backbone | Precision | Recall | F-measure | FPS | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| FAST-T-448 |TextNet-T |   86.5   |  77.2  |  81.6  | 152.8 | [config](tt/fast_tiny_tt_448_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_tt_448_finetune_ic17mlt.pth) \| [log](../../logs/tt/fast_tiny_tt_448_finetune_ic17mlt.txt)     |
| FAST-T-512 |TextNet-T |   87.3   |  80.0  |  83.5  | 131.1 | [config](tt/fast_tiny_tt_512_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_tt_512_finetune_ic17mlt.pth) \| [log](../../logs/tt/fast_tiny_tt_512_finetune_ic17mlt.txt)     |
| FAST-T-640 |TextNet-T |   87.1   |  81.4  |  84.2  | 95.5  | [config](tt/fast_tiny_tt_640_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_tt_640_finetune_ic17mlt.pth) \| [log](../../logs/tt/fast_tiny_tt_640_finetune_ic17mlt.txt)     |
| FAST-S-512 |TextNet-S |   88.3   |  81.7  |  84.9  | 115.5 | [config](tt/fast_small_tt_512_finetune_ic17mlt.py) |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_tt_512_finetune_ic17mlt.pth) \| [log](../../logs/tt/fast_small_tt_512_finetune_ic17mlt.txt)   |
| FAST-S-640 |TextNet-S |   89.1   |  81.9  |  85.4  | 85.3  | [config](tt/fast_small_tt_640_finetune_ic17mlt.py) |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_tt_640_finetune_ic17mlt.pth) \| [log](../../logs/tt/fast_small_tt_640_finetune_ic17mlt.txt)   |
| FAST-B-512 |TextNet-B |   89.6   |  82.4  |  85.8  | 93.2  | [config](tt/fast_base_tt_512_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_tt_512_finetune_ic17mlt.pth) \| [log](../../logs/tt/fast_base_tt_512_finetune_ic17mlt.txt)     |
| FAST-B-640 |TextNet-B |   89.9   |  83.2  |  86.4  | 67.5  | [config](tt/fast_base_tt_640_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_tt_640_finetune_ic17mlt.pth) \| [log](../../logs/tt/fast_base_tt_640_finetune_ic17mlt.txt)     |
| FAST-B-800 |TextNet-B |   90.0   |  85.2  |  87.5  | 46.0  | [config](tt/fast_base_tt_800_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_tt_800_finetune_ic17mlt.pth) \| [log](../../logs/tt/fast_base_tt_800_finetune_ic17mlt.txt)     |

**Results on CTW1500**

| Method | Backbone | Precision | Recall | F-measure | FPS | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| FAST-T-512 | TextNet-T  |  85.5  |  77.9  |  81.5   | 129.1 | [config](ctw/fast_tiny_ctw_512_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_ctw_512_finetune_ic17mlt.pth) \| [log](../../logs/ctw/fast_tiny_ctw_512_finetune_ic17mlt.txt) |
| FAST-S-512 | TextNet-S  |  85.6  | 78.7 | 82.0  | 112.9  | [config](ctw/fast_small_ctw_512_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_ctw_512_finetune_ic17mlt.pth) \| [log](../../logs/ctw/fast_small_ctw_512_finetune_ic17mlt.txt) |
| FAST-B-512 | TextNet-B  |  85.7  | 80.2 | 82.9  | 92.6  | [config](ctw/fast_base_ctw_512_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ctw_512_finetune_ic17mlt.pth) \| [log](../../logs/ctw/fast_base_ctw_512_finetune_ic17mlt.txt) |
| FAST-B-640 | TextNet-B  |  87.8  | 80.9 | 84.2  | 66.5  | [config](ctw/fast_base_ctw_640_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ctw_640_finetune_ic17mlt.pth) \| [log](../../logs/ctw/fast_base_ctw_640_finetune_ic17mlt.txt) |

**Results on ICDAR 2015**

| Method | Backbone | Precision  | Recall  | F-measure  | FPS | Config | Download |
| :-: | :-: |:-: | :-: | :-: | :-: | :-: | :-: |
| FAST-T-736  | TextNet-T  |    86.0       |   77.9   |    81.7    | 60.9 | [config](ic15/fast_tiny_ic15_736_finetune_ic17mlt.py)  | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_ic15_736_finetune_ic17mlt.pth) \| [log](../../logs/ic15/fast_tiny_ic15_736_finetune_ic17mlt.txt)   |
| FAST-S-736  | TextNet-S  |    86.3       |   79.8   |    82.9    | 53.9 | [config](ic15/fast_small_ic15_736_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_ic15_736_finetune_ic17mlt.pth) \| [log](../../logs/ic15/fast_small_ic15_736_finetune_ic17mlt.txt) |
| FAST-B-736  | TextNet-B  |    88.0       |   81.7   |    84.7    | 42.7 | [config](ic15/fast_base_ic15_736_finetune_ic17mlt.py)  | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ic15_736_finetune_ic17mlt.pth) \| [log](../../logs/ic15/fast_base_ic15_736_finetune_ic17mlt.txt)   |
| FAST-B-896  | TextNet-B  |    89.2       |   83.6   |    86.3    | 31.8 | [config](ic15/fast_base_ic15_896_finetune_ic17mlt.py)  | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ic15_896_finetune_ic17mlt.pth) \| [log](../../logs/ic15/fast_base_ic15_896_finetune_ic17mlt.txt)   |
| FAST-B-1280 | TextNet-B  |    89.7       |   84.6   |    87.1    | 15.7 | [config](ic15/fast_base_ic15_1280_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ic15_1280_finetune_ic17mlt.pth) \| [log](../../logs/ic15/fast_base_ic15_1280_finetune_ic17mlt.txt) |


**Results on MSRA-TD500**

| Method | Backbone | Precision | Recall | F-measure | FPS | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| FAST-T-512 | TextNet-T  |   91.1    |  78.8  |  84.5   | 137.2 | [config](msra/fast_tiny_msra_512_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_msra_512_finetune_ic17mlt.pth) \| [log](../../logs/msra/fast_tiny_msra_512_finetune_ic17mlt.txt) |
| FAST-T-736 | TextNet-T  |   88.1    |  81.9  |  84.9    | 79.6  | [config](msra/fast_tiny_msra_736_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_msra_736_finetune_ic17mlt.pth) \| [log](../../logs/msra/fast_tiny_msra_736_finetune_ic17mlt.txt) |
| FAST-S-736 | TextNet-S  |   91.6    |  81.7  |  86.4   | 72.0  | [config](msra/fast_small_msra_736_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_msra_736_finetune_ic17mlt.pth) \| [log](../../logs/msra/fast_small_msra_736_finetune_ic17mlt.txt) |
| FAST-B-736 | TextNet-B  |   92.1    |  83.0  |  87.3   | 56.8  | [config](msra/fast_base_msra_736_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_msra_736_finetune_ic17mlt.pth) \| [log](../../logs/msra/fast_base_msra_736_finetune_ic17mlt.txt) |



