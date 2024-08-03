## OMG Model Research

Our goal is to solve multiple fundamental visual perception, visual reasoning, and multi-modal large langauge tasks using **one** model, which minimize handcraft designs and maximize the functionality and performance 
in one shot.


### Short Introduction of OMG-LLaVA, [arxiv](https://arxiv.org/abs/2406.19389), [Project Page](https://lxtgh.github.io/project/omg_llava/), [Introduction by Fahd Mirza](https://www.youtube.com/watch?v=A4CWwgrxvSE)
  <p align="center">
    <a href='https://arxiv.org/abs/2406.19389'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'> </a>
    <a href='https://lxtgh.github.io/project/omg_llava/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'> </a>
    <a href='https://huggingface.co/zhangtao-whu/OMG-LLaVA' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Huggingface%20Model-8A2BE2' alt='Project Page'> </a>
    <a href="https://huggingface.co/spaces/LXT/OMG_LLaVA">
    <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-App-blue' alt='HuggingFace Model'> </a>
    <a href='[https://huggingface.co/zhangtao-whu/OMG-LLaVA/tree/main](https://73ebf9f4d6b8376505.gradio.live/)' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Gradio%20-Demo-8A2BE2' alt='Gradio'> </a>
  </p>

We present OMG-LLaVA, a new and elegant framework combining powerful pixel-level vision understanding with reasoning abilities. 
It can accept various visual and text prompts for flexible user interaction. Specifically, we use a universal segmentation method as the visual encoder, integrating image information, perception priors, and visual prompts into visual tokens provided to the LLM.
The LLM is responsible for understanding the user's text instructions and providing text responses and pixel-level segmentation results based on the visual information. 

OMG-LLaVA achieves image-level, object-level, and pixel-level reasoning and understanding in a single model, matching or surpassing the performance of specialized methods on multiple benchmarks. 
Rather than using LLM to connect each specialist, our work aims at end-to-end training on one encoder, one decoder, and one LLM.

### Short Introduction of OMG-Seg, [arxiv](https://arxiv.org/abs/2401.10229), [Project Page](https://lxtgh.github.io/project/omg_seg/), [Report By viso.ai](https://viso.ai/computer-vision/omg-seg/)
  <p align="center">
    <a href='https://arxiv.org/abs/2401.10229'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'> </a>
    <a href='https://lxtgh.github.io/project/omg_seg/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'> </a>
    <a href='https://huggingface.co/LXT/OMG_Seg' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Huggingface%20Model-8A2BE2' alt='Project Page'> </a>
    <a href="https://huggingface.co/spaces/LXT/OMG_Seg">
    <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-App-blue' alt='HuggingFace Model'> </a>
  </p>
We address various segmentation tasks, each traditionally tackled by distinct or partially unified models. 
We propose OMG-Seg, One Model that is Good enough to efficiently and effectively handle all the Segmentation tasks, including image semantic, instance, and panoptic segmentation, as well as their video counterparts, open vocabulary settings, prompt-driven, interactive segmentation like SAM, and video object segmentation.
To our knowledge, this is the first model to fill all these tasks in one model and achieve good enough performance.

We show that OMG-Seg, a transformer-based encoder-decoder architecture with task-specific queries and outputs, can support over ten distinct segmentation tasks and yet significantly reduce computational and parameter overhead across various tasks and datasets. 
We rigorously evaluate the inter-task influences and correlations during co-training. Both the code and models will be publicly available.

Short introduction on VALSE of OMG-Seg with other SAM-like works, can be found [here](https://www.bilibili.com/video/BV1PZ421b7U7/?spm_id_from=333.337.search-card.all.click&vd_source=6bb672e5bcff6f43a998d1ba30743967), in Chinese.


## News !!

- 🔥2024-6-28, Release OMG-LLaVA test code and ckpt (7B) models, will be released soon. 
- 🔥2024-4-06, Update the model trained with only one machine and demo scripts.
- 🔥2024-3-18, Training Code of OMG-Seg are released !! Stronger Performance using Object-365-instance segmentation pre-train !!
- 🔥2024-2-26, OMG-Seg is accepted by CVPR-2024 !!
- 🔥2024-1-19, Models and Test Code are released !!


## Key Features of OMG-LLaVA

### $\color{#2F6EBA}{Bridge\ Image-level\, Object-level\, Pixel-level\, Reasoning\ and\ Understanding\ }$ 

- One model to perform image level, object level, pixel level understanding and reasoning.
- A new view for solving multiple referring segmentation, localization, grounding, and captioning tasks using only one encoder, one decoder and one LLMs.

### $\color{#2F6EBA}{The\ First\ OpenSourced\ Universal\ Understanding\ and\ Reasoning\ Codebase}$  

- Our codebase supports **joint multi dense prediction tasks co-training** in one shot.
- The first open-sourced codebase for multiple multimodal understanding tasks, including training, inference and demo.


## Key Features of OMG-Seg 

### $\color{#2F6EBA}{Universal\ Image\, Video\, Open-Vocabulary\, Segmentation\ Model}$ 

- A **new unified** solution for **over ten different segmentation tasks**: PS, IS, VSS, VIS, VPS, Open-Vocabulary Seg, and Interactive Segmentation.
- A novel unified view for solving multiple segmentation tasks in one model with extremely less parameters.

### $\color{#2F6EBA}{Good\ Enough\ Performance}$  

- OMG-Seg achieves **good enough performance** on in one shared architecture, on multiple datasets. (**only 70M trainable parameters**)

### $\color{#2F6EBA}{The\ First\ OpenSourced\ Universal\ Segmentation\ Codebase}$  

- Our codebase support **joint image/video/multi-dataset co-training**.
- The first open-sourced codebase, including training, inference and demo.

### $\color{#2F6EBA}{Easy\ \ Followed\ By\ Academic\ Lab}$  

- OMG-Seg can be reproduced by only **one 32GB V100 or 40GB A100 machine**, which can be followed by Academic Labs.



## To-Do Lists 

- Add more easy-used tutorial. ()
- Release OMG-LLaVA Models. (Done)
- Release OMG-Seg Strong Models. (Done)
- Release OMG-Seg training code. (Done)
- Support HuggingFace. (Done)


## How to use this Codebase

For OMG-Seg, please see the [OMG_Seg_README.md](./OMG_Seg_README.md)

For OMG-LLaVA, please see the [OMG_LLaVA_README.md](./omg_llava/OMG_LLaVA_README.md)


## Citation

If you think our codebases and works are useful for your research, please consider referring us:


```bibtex

@article{OMGLLaVA,
  title={OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding},
  author={Zhang, Tao and Li, Xiangtai and Fei, Hao and Yuan, Haobo and Wu, Shengqiong and Ji, Shunping and Chen, Change Loy and Yan, Shuicheng},
  journal={arXiv preprint},
  year={2024}
}

@inproceedings{OMGSeg,
  title={OMG-Seg: Is one model good enough for all segmentation?},
  author={Li, Xiangtai and Yuan, Haobo and Li, Wei and Ding, Henghui and Wu, Size and Zhang, Wenwei and Li, Yining and Chen, Kai and Loy, Chen Change},
  booktitle={CVPR},
  year={2024}
}

```

## License

OMG-Seg follows the S-Lab [LICENSE](LICENSE).

OMG-LLaVA follows the [Apache-2.0 license](https://github.com/haotian-liu/LLaVA?tab=Apache-2.0-1-ov-file), for the respect of both [LLaVA](https://github.com/haotian-liu/LLaVA) and [XTuner](https://github.com/InternLM/xtuner) codebase.
