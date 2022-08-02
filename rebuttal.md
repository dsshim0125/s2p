# General Comments

We sincerely thank all the reviewers for providing constructive comments.

1. We provide a correction for common misunderstandings of the reviewers to clarify our methodology.
**S2P itself is not an RL algorithm.** Rather, it is an image synthesis algorithm for data augmentation. S2P only generates images from states in the extended distribution and the previous image. The synthesized image outputs can be used as additional data during training for any image-based offline RL algorithms.

2. If you don’t mind, please check the Appendix in the supplementary files. Most of the answers are already provided in it.

3. We have uploaded a new version of the manuscript based on reviewers’ comments and minor corrections.

# Response to Reviewer i7hg

## Q1 : Standard deviation of scores in tables.

We attached the standard deviation of the tables.

Table 1 : https://drive.google.com/file/d/1OnymWPD1R_r_1fB3zcE0evagX9oyGSFr/view?usp=sharing

Table 2 : https://drive.google.com/file/d/1hY5wW63VRHKd8UH0W6K6UaDwn6yU_cpD/view?usp=sharing

Table 3 : https://drive.google.com/file/d/1YmdM01Q_bNG6bna4qphX2UCB8ojJ9kfU/view?usp=sharing

Table 4 : https://drive.google.com/file/d/11M_etepameN7zoU2fmpmh1_ECTUrjk-H/view?usp=sharing

## Q2: S2P vs Naive image augmentation
S2P outperforms the previous most frequently used & effective image augmentation strategies, REFLECT&CROP, for image RL. Please refer to Table 10 in Appendix. (This strategy’s effectiveness compared to other augmentation strategies like convolution, color-jittering, etc, is already proved experimentally in several model-free image RL papers such as CURL[1], RAD[2], DrQ[3]). 

It is because a simple augmentation strategy is just image manipulation rather than broadening the dataset distribution’s support as mentioned in our work. 

[1] : Srinivas, Aravind, Michael Laskin, and Pieter Abbeel. "Curl: Contrastive unsupervised representations for reinforcement learning." arXiv preprint arXiv:2004.04136 (2020).

[2] Laskin, Misha, et al. "Reinforcement learning with augmented data." Advances in neural information processing systems 33 (2020): 19884-19895.

[3] Yarats, Denis, Ilya Kostrikov, and Rob Fergus. "Image augmentation is all you need: Regularizing deep reinforcement learning from pixels." International Conference on Learning Representations. 2020.

## Q3: S2P is not needed with random policy?
Pardon. We are sorry that we cannot fully understand your question. To clarify your intention, we provide two different answers. If both answers are not satisfactory, please let us know. 

Q3.1: If your comment means that the S2P is some kind of policy or RL algorithm, so you refer to the “S2P can be not needed”, Please check the general comments above first.

Q3.2:  If your intention was “We do not need S2P because we can improve performance with random policy”, S2P is needed anyway to leverage random or perturbed policy in image RL by synthesizing images from the randomly visited state.


## Q4. How much data do you need to train S2P model?  
The same amount of data used for training the offline RL algorithm is used to train S2P, which is 50k in this work.

## Q5. How can you evaluate the quality of synthesized images?
We evaluate the image fidelity of our S2P on 4 different metrics (FID, LPIPS, PSNR, SSIM)  in Table 6 (Appendix) which are widely adopted to validate the quality of generated images. Our S2P outperforms Dreamer in all the metrics on average.


We appreciate all your efforts during the review process. We believe to have addressed all the comments. If this response is not sufficient enough to raise the score, please do not hesitate to let us know.


# Response to Reviewer bQSP 


## Q1. Explicitly mention limitation 
We admit that training S2P requires an assumption that state and its paired images are needed, so we explicitly mention it in the manuscript. Please check the newly uploaded manuscript (340-343).

## Q2. Image to state (P2S) vs state to image (S2P)
Thank you for your interesting idea, and we additionally experiment with the image-to-state model (P2S) compared to S2P.
We train the state estimator (P2S) in two different ways. 1) train state estimator from a single image 2) train state estimator from three consecutive images. The difference is that we induce the latter one to capture the velocity information from the stacked images as previous image-based RL methods do. The state estimator network consists of CNN layers followed by MLP layers and it is trained by MSE loss between predicted states and ground truth states. Then, we train a state-based offline RL with the augmented dataset, where the states in the dataset are predicted by the trained state estimator network. And we evaluate the performance by sampling the action from the state-based policy, where the state input for the policy is predicted from the image given by the environment at every timestep. The results are shown in the following Table.

https://drive.google.com/file/d/16JSTfPjjQhNNsksagCpgphGxpSsv81Vs/view?usp=sharing

As we expected, the state estimator from stacked images produces better performance compared to the state estimator from a single image. However, they cannot outperform S2P in every task, and even cannot outperform the baseline algorithm without S2P in most of the tasks. 
The reason for such performance degradation is that it is difficult to estimate the accurate state information from images that lie in a high-dimensional space. Also, even though the MSE loss is decreased on the given paired image-state dataset by overfitting, the agent observes the unseen image during evaluation, which leads to inaccurate state estimation from unseen image input, and results in drastic performance degradation.

Unlike state estimation strategy (P2S), S2P leverages multi-model inputs (state, previous image) which have similar dimensionality to the outputs, and it makes S2P synthesize accurate images extracting information from two different modalities. Also, we adopt three different loss functions (L1, GAN, perceptual) so that the combination of the losses mitigates the overfitting to the offline dataset’s distribution. 

## Q3. State and Image conditioned Dreamer vs S2P
As you thought it is a fair evaluation for Dreamer to train with the state information same as S2P, we trained the image and state concatenated Dreamer in cheetah, walker environments. Specifically, the encoder of Dreamer takes the image and state, and the decoder reconstructs the image and state. The state reconstruction loss is the same as the image reconstruction loss (likelihood maximization). Then, we augment the dataset by generating image transition data the same as +DREAMER case in Table 3 in our work (or Table 10 in the appendix). The offline RL performance of state concatenated Dreamer is shown in the following Table.

https://drive.google.com/file/d/1fDN3mS4kxAvnbf_AoP2VSq9ufrZe1JvK/view?usp=sharing

The state concatenated Dreamer shows inconsistent performance compared to the Dreamer and still produces worse performance in all the tasks compared to S2P. It is because the architecture of Dreamer itself cannot capture the multi-modal input effectively, while we design S2P for leveraging multi-modal inputs to synthesize image (MAT) and S2P leverages both state and image information for predicting the agent’s image. To visualize the difference between S2P and Dreamer in exploiting multi-modality, we mask out the state and see how the output image is affected. When we zero mask out the state in Dreamer, there is no significant difference. However, when we remove the position element of the state, S2P does not produce agents, but only reconstructs the backgrounds which contain the velocity information of the state. When we zero mask out the velocity information in S2P, the agent walks but it walks in the place without moving forward as the velocity is zero.

state concatenated dreamer- walker:
https://drive.google.com/file/d/146qw83A3S-m1-MHTE8xxLeyp8jeF1Bsr/view?usp=sharing

state concatenated dreamer- cheetah:
https://drive.google.com/file/d/1cMB903IUdApNRJE0Tf09Xz2PHItNdizh/view?usp=sharing
 
masking ablation : 
https://drive.google.com/file/d/1-nnZCoFpbMKGVllnQc6LIdl7Bntgk-F2/view?usp=sharing


## Q4. Societal Impact
This work inherits the potential negative societal impacts of reinforcement learning. We do not anticipate any additional negative impacts that are unique to this work.

We appreciate all your efforts during the review process. We believe to have addressed all the comments. If this response is not sufficient enough to raise the score, please do not hesitate to let us know.


# Response to Reviewer h9Ka 

## Q1. S2P with noisy or unreliable state transition
As you mentioned, we expect S2P not to perform reasonably where the state transition is not reliable or noisy (e.g. extremely complex dynamics of the agent, or real-world data with sensor noise). Nevertheless, we believe that the state transition is still much more refined and reliable information compared to pixel-level transition data, and therefore, leveraging S2P with noisy state transition is still a good choice for data augmentation in offline image RL settings.

## Q2. The size of the dataset for training,  Mixing ratio
First of all, we would like to make clear that we did not use any additional ground truth data when we obtain all the offline RL results of +S2P experiments in our work including Table 1. We only use the originally given fixed 50k dataset to train S2P, and any additional data used for obtaining the results in +S2P experiments in our work is generated by the S2P (It is not ground truth data). Therefore, it could be considered fair even though the total number of transitions is different when we address the data augmentation problems.

Specifically, the left column of each baseline(ex: IQL, CQL, SLAC-off) in Table 1 is the trained results with the 50k dataset, and the right column (ex: IQL+S2P, CQL+S2P, SLAC-off+S2P) is the trained results with the original 50k + generated 50k dataset by S2P (totally 100k). And, as we mentioned in our work, the scores in the parenthesis are the trained results with the 100k ground truth dataset (it does not contain generated data by S2P, and is just additionally collected for reference to match the same amount). We validate through this experiment that RL agents trained with augmented data (ground truth 50k + S2P 50k) produces the comparable or even better performance compared to data which has the same transition number from ground truth (ground truth 100k).


Therefore, we think fixing the entire transition of the data is not a proper experiment setting for evaluating the performance according to the ratio of ground truth and synthesized frames.
Rather, we fix the number of the ground truth and change the number of the synthesized images to figure out how much additional data effects the RL performance.
The results are shown in the following table.
<!--We also additionally experimented with the mixing ratios of the datasets. The results are shown in the following Table.-->

https://drive.google.com/file/d/14J8vgL9wYyrr7Z5s8oFeckTOGO9Gm4MX/view?usp=sharing

We use the originally given 50k offline dataset, and vary the ratio of the generated dataset by S2P. (That is, 5:N means that the ratio of the originally given offline dataset is 5, generated dataset by S2P is N). Except for the already enough successful baseline such as cheetah-CQL, our S2P overall increases the offline RL performance, regardless of the ratio, in all environments and baselines. Some slightly decreased performances of 5:7.5 compared to the 5:5 or 5:2.5 seems to be the random policy’s effect (As mentioned in our work, the random policy is used for mixed-level datasets), because too much randomness in the dataset could make the overall dataset’s expertise lower.


## Q3. Typos & Reference 
We fix typos and cite missing references in the newly uploaded manuscript file.

## Q4. Ablation on state masking: 
We appreciate your brilliant idea, and we additionally experiment with ablation studies that mask out the state information.

https://drive.google.com/file/d/1-nnZCoFpbMKGVllnQc6LIdl7Bntgk-F2/view?usp=sharing

We mask out either the position or velocity of the state and each presents different aspects. When we mask out the position of the state, S2P cannot capture the posture of the agent, but it captures the velocity by generating the change of the backgrounds (checkerboard ground) as we expected. When the velocity of the state is masked out, S2P still recovers the posture of the agent, but it walks in place without moving forward due to the missing velocity.
Without the previous image, S2P recovers the posture of the agent plausibly, but as it cannot access the previous image, the background cannot be reconstructed.

## Q5. Ablation on the loss function
All the loss combined shows better qualitative results compared to the sole L1 loss as shown in the figure below link. 
 
https://drive.google.com/file/d/1Ef7ahMfcikyJhnpX7bbWKOfQKtucKxZd/view?usp=sharing

We attribute such image quality to the adversarial GAN loss with perceptual loss as GAN generates photorealistic images and perceptual loss is known as synthesizing the details of the image [1] such as hair or fur, in this case, the tip of the limb.


[1] Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." European conference on computer vision. Springer, Cham, 2016.

We appreciate all your efforts during the review process. We believe to have addressed all the comments. If this response is not sufficient enough to raise the score, please do not hesitate to let us know.


# Response to Reviewer WCLw 

## Q1. MAT is not novel which is similar to SPADE [1] and ManiGAN [2]: 
We respectfully disagree. As we mentioned in the manuscript (59-61, 87-91), our proposed MAT fuses multi-modal signals in estimating the modulation parameters in Adaptive Instance Normalization (AdaIN) whereas previous methods (SPADE [1], ManiGAN [2]) only utilizes a single domain signal (semantic map, reference image) for style/domain transfer.

Modalities in style transfer module (AdaIN)

SPADE input: random variable z, modulation: semantic map

ManiGAN input: text, modulation: image

S2P input: state and image, modulation: state and image

If there exists other prior literature that also deals with multi-modal input and modulation, would you mind referring to them? We will be glad to discuss the difference between S2P.

[1] Li B, Qi X, Lukasiewicz T, et al. Manigan: Text-guided image manipulation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 7880-7889.

[2] Park T, Liu M Y, Wang T C, et al. Semantic image synthesis with spatially-adaptive normalization[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 2337-2346.

## Q2 : Why is high-quality and accurate synthesis image generation important than data generation in latent space?
The Dreamer exactly does the data generation in latent space by deploying the latent dynamics. However, the ELBO-based method such as Dreamer only can encode the seen images, which means the latent feature involves the information of the seen images. That is, if we explore the latent space, different from S2P (state space), the visited novel latent feature is not guaranteed to represent the novel state(image) in the original domain. Therefore, it is important to distill as much information as possible to the image encoder of the policy, and critic networks by producing diverse image data distribution despite causing an additional learning process to encode the synthesized images. 

Furthermore, the advantages of our S2P could be further validated by the generalization test in Section 4.5 for answering the image generation capability at the newly visited states by the learned state dynamics model. For example, the running walker's images could be augmented even though we only have the state transition data and only a single image at t = 0. The Dreamer-like method that generates in latent space cannot adapt to this new task because it has not ever seen the running walker's images. 

Also, we could further consider extending the Dreamer with state inputs, as requested by another reviewer (Please refer to the answer of Q3 for the reviewer "bQSP" if you are interested in). But it also has difficulty in reflecting the state information into image generation due to the lack of architectural design for multi-modal inputs.

And we experimentally show the results in Table 3 and Figure 4 in our work, and Table 10 in the appendix. Due to the aforementioned reasons, our S2P achieves much better performance increases compared to the Dreamer that generates the data in latent space.


## Q3 : Offline RL performance with SAT vs MAT
For the ablation study, we additionally trained the generator only using the SAT and tested the offline learning performance in the walker environment. The results are shown in the following Table. 

https://drive.google.com/file/d/1T5xV62voEdUM221JzI9Zz-SoF6W8X5DO/view?usp=sharing

As mentioned in Figure 5, misalignment of the checkerboard affects the recognition of the agent’s dynamics and it results in performance degradation.

## Q4. Performance comparison with COMBO, ROMI
The referred paper utilizes forward or reverse dynamics for data generation. However, ROMI does not extend the idea into the image domain, and COMBO experimented in the image domain, which uses a latent dynamics model based on ELBO. As far as we know, augmenting the image transition data at COMBO is performed by generating the future image from latent space, and it would still have similar problems observed in Dreamer, because they do not consider the multi-modal inputs (Answered in Q2). The walker environment is also experimented with in COMBO, and it results in a 57.7-76.4 score (Table 2 in COMBO), while our S2P achieves almost expert level (70.95-97.97) score along all baselines. 



## Q5. Why use the previous step image instead of the previous step state?
If we do not use the previous image, we cannot accurately predict the image because the sole state information cannot give the background information (e.g. checkerboard on the ground) which contains the velocity information in the image. It can be also validated as shown in Figure 5-c in our work.



## Q6. Why VGG model pretrained with ImageNet in RL tasks: 
We do not use the VGG model for RL tasks, but use them for utilizing perceptual loss in training S2P which is for the ‘image generation’ task. In image generation, it is well known that perceptual loss recovers the details of the image such as the fur of the animal, in this case, the tip of the limb [1].

[1] Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." European conference on computer vision. Springer, Cham, 2016.

## Q7. Is baseline encoder different from S2P?
S2P is not an RL algorithm, rather it is an image synthesis model. Additional training images for offline RL are generated from S2P and any image-based offline RL algorithms can take advantage of S2P. During all the experiments, we use the same baseline architecture which means that the encoders are all the same across the offline RL algorithms (IQL, CQL, SLAC-off). Therefore, it is a fair evaluation when we compare offline RL results.

## Q8. Why introducing more generated image data cannot bring distinct performance improvement as shown in Table 1? 
Pardon. If your question means that the scores in baseline +S2P (right column of each baseline offline RL algorithm in Table 1) and scores in parenthesis are not that distinct, and you understand that the scores in parenthesis are obtained by augmenting more S2P outputs, we are sorry for misleading your understanding. 

Specifically, the left column of each baseline(ex: IQL, CQL, SLAC-off) in Table 1 is the trained results with the 50k dataset, and the right column (ex: IQL+S2P, CQL+S2P, SLAC-off+S2P) is the trained results with the original 50k + generated 50k dataset by S2P (totally 100k). And, as we mentioned in our work, the scores in the parenthesis are the trained results with the 100k ground truth dataset (it does not contain generated data by S2P, and is just additionally collected for reference to match the same amount). The data augmentation techniques’ contribution is that we can get the comparable or even better performance amount of the 100k dataset despite only having the smaller size dataset (50k in this case). Thus, it could be considered fair when we address the data augmentation problems.

If your question was intended to ask for a comparison of results based on the amount of generated data by S2P compared to the originally given offline dataset(50k), we also additionally experimented with the mixing ratios of the datasets. The results are shown in the following Table. 

https://drive.google.com/file/d/14J8vgL9wYyrr7Z5s8oFeckTOGO9Gm4MX/view?usp=sharing

We use the originally given 50k offline dataset, and vary the ratio of the generated dataset by S2P. (That is, 5:N means that the ratio of the originally given offline dataset is 5, generated dataset by S2P is N). Except for the already enough successful baseline such as cheetah-CQL, our S2P overall increases the offline RL performance, regardless of the ratio, in all environments and baselines. Some slightly decreased performances of 5:7.5 compared to the 5:5 or 5:2.5 seems to be the random policy’s effect (As mentioned in our work, the random policy is used for mixed-level datasets), because too much randomness in the dataset could make the overall dataset’s expertise lower.


If we misunderstand your question, please let us know to provide proper responses (with or without experiment)


We appreciate all your efforts during the review process. We believe to have addressed all the comments. If this response is not sufficient enough to raise the score, please do not hesitate to let us know.
