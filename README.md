# Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images

This repository provides the pytorch implementatin of our ICLR 2024 work: [Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images](https://arxiv.org/abs/2303.12993).

## Abstract

Large vision-language models (VLMs) such as GPT-4 have achieved exceptional performance across various multi-modal tasks. However, the deployment of VLMs necessitates substantial energy consumption and computational resources. Once attackers maliciously induce high energy consumption and latency time (energy-latency cost) during inference  of VLMs, it will exhaust computational resources. In this paper, we explore this attack surface about availability of VLMs and aim to induce high energy-latency cost during inference of VLMs. We find that high energy-latency cost during inference of VLMs can be manipulated by maximizing the length of generated sequences. To this end, we propose verbose images, with the goal of crafting an imperceptible perturbation to induce VLMs to generate long sentences during inference. Concretely, we design three loss objectives. First, a loss is proposed to delay the occurrence of end-of-sequence (EOS) token, where EOS token is a signal for VLMs to stop generating further tokens. Moreover, an uncertainty loss and a token diversity loss are proposed to increase the uncertainty over each generated token and the diversity among all tokens of the whole generated sequence, respectively, which can break output dependency at token-level and sequence-level. Furthermore, a temporal weight adjustment algorithm is proposed, which can effectively balance these losses. Extensive experiments demonstrate that our verbose images can increase the length of generated sequences by 7.87 times and 8.56 times compared to original images on MS-COCO and ImageNet datasets, which presents potential challenges for various applications.

<div align=center>
<img src="assets/verbose_images.png" width="800" height="300" alt="Pipeline of ASD"/><br/>
</div>

## Installation

This code is tested on our local environment (python=3.9.2, cuda=11.6), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n VI python=3.9.2
```
Then, activate the environment:
```bash
conda activate VI
```

Install requirements:

```bash
pip install -e .
```

## Data Preparation

Please download MS-COCO dataset from its [official
website](https://cocodataset.org/#download) and randomly select 1,000 images.

## Verbose Images

Run the following command to generate verbose images to induce high energy-latency cost of BLIP-2.

```shell
bash scripts/run.sh
```

## Citation

```
coming soon
```

## Acknowledgements

This respository is mainly based on [LAVIS](https://github.com/salesforce/LAVIS) and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). Thanks for their wonderful works!