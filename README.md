# Face Classification based on Unknown Criteria

This repository comprises my solution to a datachallenge organised at Telecom Paris in partnership with Idemia at the end of an Advanced Machine Learning course. It contains :
* A [jupyter notebook](./Face_Classification_Unk.ipynb) detailing my classification algorithm and the choices taken to develop it (used as a report)
* [Python scripts](./scripts) used for training the best model, classifying the images and generating the attention maps
* A [presentation](./Restitution_Datachallenge.pdf) of the approach in PDF format, used to explain my reasoning to the whole promotion at the end of the challenge
* The [images](./data_challenge_Avril_2023) used as dataset for the challenge

## Context of the project

The task of this datachallenge was a simple binary classification of face images. Two specificities were added to spice it up a little bit :
* **Unknown criteria** <br>
The criteria used to assign a label `1` or `-1` to an image was hidden, and not obvious. This, in itself, does not affect the difficulty of training a classifier to solve the task. However, it calls for caution when using data augmentation techniques, as class specific information may be erased or modified in this process. Besides, it also prevents the use of multimodal models such as CLIP or Align, which are strong zero-shot classifiers but require a natural language prompt precisely describing the classes in order to achieve good results.
* **Gender neutral evaluation metric** <br>
The evaluation metric is calculated with the following formula, which makes it necessary to obtain a similar accuracy score for men and women in order to be well ranked :

$$ mean(accuracy_{men}, accuracy_{women}) - abs(accuracy_{men} - accuracy_{women})$$

The face images were all extracted from the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and programmatically altered (resolution and data augmentation techniques) to produce the train and test datasets. <br>
The **trainset** consists of **192 577 RGB images** of size 80x80 pixels, the **testset** consists of **10 000 RGB images** of size 80x80 pixels.

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
