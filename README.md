# HIU-DMTL
This is the official code for paper `Hand Image Understanding via Deep Multi-Task Learning`. The the pre-trained model will be released soon.  Thank you for your attention.

Analyzing and understanding hand information from multimedia materials like images or videos is important for many real world applications and remains active in research community. There are various works focusing on recovering hand information from single image, however, they usually solve a single task, for example, hand mask segmentation, 2D/3D hand pose estimation, or hand mesh reconstruction and perform not well in challenging scenarios. To further improve the performance of these tasks, we propose a novel Hand Image Understanding (HIU) framework to extract comprehensive information of the hand object from a single RGB image, by jointly considering the relationships between these tasks. To achieve this goal, a cascaded multitask learning (MTL) backbone is designed to estimate the 2D heat maps, to learn the segmentation mask, and to generate the intermediate 3D information encoding, followed by a coarse-to-fine learning paradigm and a self-supervised learning strategy

### Demos.
We present three videos to illustrate the HIU-DMTL framework, including [_example speech_](https://youtu.be/ZtVAPvVcmZ8), [_example dance_](https://youtu.be/tFZiHM8tq3E), and [_in the wild video_](https://youtu.be/5eTXbzqrBYE).



### The new dataset.
The [following link](https://pan.baidu.com/s/1HHfj9nqb27YBJZ0dCgo8_g) (提取码: utz8 复制这段内容后打开百度网盘手机App，操作更方便哦) will be disabled after Jan.8 2022. Since then, to obtain the well-defined dataset, please feel free to drop me a email (1025679612 at qq dot com). 
Additionally, the hiu_dmtl_data.zip has been encrypted, please email to me to get the password.

### Citation
If you use this code/dataset for your research, please cite:
```
@inproceedings{zhang2021hand,
  title={Hand Image Understanding via Deep Multi-Task Learning},
  author={Zhang, Xiong and Huang, Hongsheng and Tan, Jianchao and Xu, Hongmin and Yang, Cheng and Peng, Guozhu and Wang, Lei and Liu, Ji},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={11281--11292},
  year={2021}
}

@inproceedings{zhang2019end,
  title={End-to-end hand mesh recovery from a monocular rgb image},
  author={Zhang, Xiong and Li, Qiang and Mo, Hong and Zhang, Wenbo and Zheng, Wen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={2354--2364},
  year={2019}
}
```
