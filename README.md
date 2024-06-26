
<p align="center">
  <h1 align="center">OV-VG: A Benchmark for Open-Vocabulary Visual Grounding</h1>
  <p align="center">
    <br />
    <strong>Chunlei Wang</strong></a>
    ·
    <strong>Wenquan Feng</strong></a>
    ·
    <a href="https://sites.google.com/view/guangliangcheng"><strong>Guangliang Cheng</strong></a>
    ·
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li</strong></a>
    ·
    <a href="https://cv-shuchanglyu.github.io/EnHome.html"><strong>Shuchang Lyu</strong></a>
    ·
    <strong>Binghao Liu</strong></a>
    <br />
    ·
    <strong>Lijiang Chen</strong></a>
    ·
    <strong>Qi Zhao</strong></a>
    <br />
  </p>

![teaser](./images/problem_setting.png)

## Highlight!!!!

OV-VG: Open-vocabulary Visual Grounding

## Abstract

Open-vocabulary learning has emerged as a cutting-edge research area, particularly in light of the widespread adoption of vision-based foundational models. Its primary objective is to comprehend novel concepts that are not encompassed within a predefined vocabulary. One key facet of this endeavor is Visual Grounding (VG), which entails locating a specific region within an image based on a corresponding language description. While current foundational models excel at various visual language tasks, there's a noticeable absence of models specifically tailored for open-vocabulary visual grounding (OV-VG). This research endeavor introduces novel and challenging OV tasks, namely Open-Vocabulary Visual Grounding (OV-VG) and Open-Vocabulary Phrase Localization (OV-PL). The overarching aim is to establish connections between language descriptions and the localization of novel objects. To facilitate this, we have curated a comprehensive annotated benchmark, encompassing 7,272 OV-VG images (comprising 10,000 instances) and 1,000 OV-PL images. In our pursuit of addressing these challenges, we delved into various baseline methodologies rooted in existing open-vocabulary object detection (OV-D), VG, and phrase localization (PL) frameworks. Surprisingly, we discovered that state-of-the-art (SOTA) methods often falter in diverse scenarios. Consequently, we developed a novel framework that integrates two critical components: Text-Image Query Selection (TIQS) and Language-Guided Feature Attention (LGFA). These modules are designed to bolster the recognition of novel categories and enhance the alignment between visual and linguistic information. Extensive experiments demonstrate the efficacy of our proposed framework, which consistently attains SOTA performance across the OV-VG task. Additionally, ablation studies provide further evidence of the effectiveness of our innovative models.

![teaser](./images/method.png)

## TODO
- [x] Release demo
- [x] Release checkpoints
- [x] Release DATASET
- [ ] Release training and inference codes

## Install
```bash
$ git clone https://github.com/cv516Buaa/OV-VG
$ cd OV-VG
$ pip install -r requirements.txt
$ cd demo
$ python demo.py
```
## Checkpoints
* `OV-VG`:  | [Baidu Drive(pw: ovvg)](https://pan.baidu.com/s/1IHWS8_4yzR0SWvBp7qp9xw). |  [Google Drive](https://drive.google.com/file/d/1BhD1oWXddr6sb6SJdU0cRIpW91gfeDiU/view?usp=drive_link) |

## Dataset
* `OV-VG`:  | [Baidu Drive(pw: ovvg)](https://pan.baidu.com/s/1VfrtFyVZrMtgFITfLwKOGg). |  [Google Drive](https://drive.google.com/file/d/1nMv1a-afphFiO5yascga4NuIoa1ZYZXZ/view?usp=drive_link) |
* `OV-PL`:  | [Baidu Drive(pw: ovvg)](https://pan.baidu.com/s/1K42olNe-OOS_crymvISgCg). |  [Google Drive](https://drive.google.com/file/d/1KUZMiaGEevkROX5nRD3nCYNNtWmU788p/view?usp=drive_link) |

## Visualization
![teaser](./images/visual_results_1.png)

![teaser](./images/visual_results_2.png)

## Citation

- https://arxiv.org/abs/2310.14374 

If you have any question, please discuss with me by sending email to wcl_buaa@buaa.edu.cn.

If you find this code useful please cite:
```
@article{wang2023ov,
  title={OV-VG: A Benchmark for Open-Vocabulary Visual Grounding},
  author={Wang, Chunlei and Feng, Wenquan and Li, Xiangtai and Cheng, Guangliang and Lyu, Shuchang and Liu, Binghao and Chen, Lijiang and Zhao, Qi},
  journal={arXiv preprint arXiv:2310.14374},
  year={2023}
}
```
