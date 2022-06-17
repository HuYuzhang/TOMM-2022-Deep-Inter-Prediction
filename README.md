## Deep Inter Prediction with Error-Corrected Auto-Regressive Network for Video Coding (TOMM)

[Yuzhang Hu](https://huyuzhang.github.io/), [Wenhan Yang](https://flyywh.github.io/index.html), [Jiaying Liu](http://www.icst.pku.edu.cn/struct/people/liujiaying.html), and Zongming Guo

[[Paper Link]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Self-Learning_Video_Rain_Streak_Removal_When_Cyclic_Consistency_Meets_Temporal_CVPR_2020_paper.pdf) [[Project Page]](https://github.com/flyywh/CVPR-2020-Self-Rain-Removal) [[Slides]]()(TBA)[[Video]]()(TBA) (CVPR'2020 Poster)

### Abstract

Modern codecs remove temporal redundancy of a video via inter prediction, i.e. searching previously coded frames for similar blocks and storing motion vectors to save bit-rates.

However, existing codecs adopt block-level motion estimation, where a block is regressed by reference blocks linearly and is doomed to fail to deal with non-linear motions.

In this paper, we generate virtual reference frames with previously reconstructed frames via deep networks to offer an additional candidate, which is not constrained to linear motion structure and further significantly improves coding efficiency.

More specifically, we propose a novel deep Auto-Regressive Moving-Average (ARMA) model, Error-Corrected Auto-Regressive Network (ECAR-Net), equipped with the powers of the conventional statistic ARMA models and deep networks jointly for reference frame prediction.

Similar to conventional ARMA models, the ECAR-Net consists of two stages: Auto-Regression (AR) stage and Error-Correction (EC) stage, where the first part predicts the signal at the current time-step based on previously reconstructed frames while the second one compensates for the output of the AR stage to obtain finer details. Different from the statistic AR models only focusing on short-term temporal dependency, the AR model of our ECAR-Net is further injected with the long-term dynamics mechanism, where long temporal information is utilized to help predict motions more accurately. Furthermore, ECAR-Net works in a configuration-adaptive way, i.e. using different dynamics and error definitions for the Low Delay B and Random Access configurations, which helps improve the adaptivity and generality in diverse coding scenarios. With the well-designed network, our method surpasses HEVC on average 5.0% and 6.6% BD-rate saving for the luma component under the Low Delay B and Random Access configurations and also obtains on average 1.54% BD-rate saving over VVC. Furthermore, ECAR-Net works in a configuration-adaptive way, i.e. using different dynamics and error definitions for the Low Delay B and Random Access configurations, which helps improve the adaptivity and generality in diverse coding scenarios.

#### If you find the resource useful, please cite the following :- )

```
@article{Hu_2022_TOMM,
author = {Yuzhang Hu and Yang, Wenhan and Liu, Jiaying and Zongming Guo},
title = {Deep Inter Prediction with Error-Corrected Auto-Regressive Network for Video Coding},
booktitle = {ACM Transactions on Multimedia Computing Communications and Applications},
month = {June},
year = {2022}
}
```

## Contact

If you have questions, you can contact `yuzhanghu@pku.edu.cn`.