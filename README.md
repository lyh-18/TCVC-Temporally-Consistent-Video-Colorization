# TCVC-Temporally-Consistent-Video-Colorization
## Temporally Consistent Video Colorization with Deep Feature Propagation and Self-regularization Learning.  
[[paper]](https://arxiv.org/pdf/2110.04562.pdf) [[demo]](https://www.youtube.com/watch?v=c7dczMs-olE&t=22s)

### Brief Introduction
Video colorization is a challenging and highly ill-posed problem. Although recent years have witnessed remarkable progress in single image colorization, there is relatively less research effort on video colorization and existing methods always suffer from severe flickering artifacts (temporal inconsistency) or unsatisfying colorization performance. We address this problem from a new perspective, by jointly considering colorization and temporal consistency in a unified framework. Specifically, we propose a novel temporally consistent video colorization framework (TCVC). TCVC effectively propagates frame-level deep features in a bidirectional way to enhance the temporal consistency of colorization. Furthermore, TCVC introduces a self-regularization learning (SRL) scheme to minimize the prediction difference obtained with different time steps. SRL does not require any ground-truth color videos for training and can further improve temporal consistency. Experiments demonstrate that our method can not only obtain visually pleasing colorized video, but also achieve clearly better temporal consistency than state-of-the-art methods.

## Visual Comparison
![visual_comparison1](compare1.png)  
![visual_comparison2](compare2.png)  
Image-based colorization method, e.g. InsColor (CVPR2021), tends to bring about severe flickering artifacts with inconsistent colors (highlighted in green rectangles). The colorization effect of video-based method FAVC (CVPR2019) is not satisfactory. Instead, our method can achieve good temporal consistency while maintaining excellent colorization performance.
  
![visual_comparison3](compare3.png)  
Video demo and comparison can be found at this link: [[demo]](https://www.youtube.com/watch?v=c7dczMs-olE&t=22s)

## Method
![framework](framework.png)  
The proposed TCVC framework (take N=4 for example). The anchor frame branch colorizes the two anchor frames and extracts the deep features for propagation. With bidirectional deep feature propagation, the internal frame features are all generated from anchor frames, which ensures the temporal consistency in high-dimensional feature space.