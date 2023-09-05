# Face Classification based on Unknown Criteria

This repository comprises my solution to a datachallenge organised at Telecom Paris at the end of an Advanced Machine Learning course. It contains :
* A [jupyter notebook](./Face_Classification_Unk.ipynb) detailing my classification algorithm and the choices taken to develop it (used as a report)
* [Python scripts](./scripts) used for training the best model, classifying the images and generating the attention maps
* A [presentation](./Restitution_Datachallenge.pdf) of the approach in PDF format, used to explain my reasoning to the whole promotion at the end of the challenge
* The [images](./data_challenge_Avril_2023) used as dataset for the challenge

## Context of the project

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Key difficulties

## Results
![](./images/screenshot_results.jpg)

## References
1. **DINO** <br>
Paper : Mathilde Caron & al. *Emerging Properties in Self-Supervised Vision Transformers*, [arXiv:2104.14294](https://arxiv.org/pdf/2104.14294.pdf), April 2021 <br>
Code : https://github.com/facebookresearch/dino
2. **SING** <br>
Paper : Adrien Courtois & al. *SING: A Plug-and-Play DNN Learning Technique*, [arXiv:2305.15997](https://arxiv.org/pdf/2305.15997.pdf), May 2023 <br>
Code : https://github.com/adriencourtois/sing
3. **FFCV** <br>
Paper : Guillaume Leclerc & al. *FFCV: Accelerating Training by Removing Data Bottlenecks*, [arXiv:2306.12517](https://arxiv.org/pdf/2306.12517.pdf), June 2023 <br>
Code : https://ffcv.io/
