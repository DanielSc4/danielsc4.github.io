---
layout: post
title: "Let the Models Respond: Interpreting the Detoxification process of LMs"
date: 2023-07-12 11:59:00-0400
description: Journal to keep track of work during internship @RUG
categories: journal
giscus_comments: false
related_posts: false
toc:
  beginning: true
---


#### **Framing**

🚨 This blogpost contains examples which are offensive in nature.


This research project was carried out by [me 👋🏼](https://www.danielsc4.it/) during the internship period at the [Computational Linguistics Research Lab](https://www.rug.nl/research/clcg/research/cl/?lang=en) at the [University of Groningen](https://www.rug.nl/). Currently, the work is still in progress and nearing completion. The results and status of the work do not represent the final state of the research.

The work is supervised by:
- [Gabriele Sarti](https://gsarti.com/), PhD student @ University of Groningen
- [Malvina Nissim](https://www.rug.nl/staff/m.nissim/), Full professor @ University of Groningen
- [Elisabetta Fersini](https://en.unimib.it/elisabetta-fersini), Associate professor @ University of Milano - Bicocca

<hr>

<br/>
## **📜 Abstract**

**Language Models** (LMs) represent complex systems that are difficult to manage and deploy safely. For this reason, various techniques have been proposed over time with the aim of detoxifying and controlling the behaviour of the models after their training process. With this in mind, this research project aims to **explore the potential of the model detoxification process**. Known techniques of *fine-tuning* and *Reinforcement Learning from Human Feedback* (RLHF) will be explored leading to less toxic models. The work also aims to **understand the detoxification process through an exploration on the interpretability of the models** themselves, having the ultimate goal of **not limiting their responses** but offering a contronarrative with respect to potentially toxic prompts.


<br/>
## **🎨 Introduction and State Of The Art**
In the recent period, LMs are observing a rise in terms of parameters, complexity and consequently results obtained that, in some cases, manage to exceed even human capabilities for specific tasks [(Radford and Narasimhan, 2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). All this power, however, comes from large amounts of data used in the pre-training phase of LMs that learn primarily from corpora extracted from the Internet, forums and social media. The large availability of text on these platforms certainly implies an ease in extracting various aspects of language useful for the learning process but brings with it issues especially relevant to the quality and content itself in the text. Indeed, it is not at all uncommon to find toxic, dangerous, privacy-compromising content or more complex phenomena such as unintended bias hidden in the text itself [(Bender et al., 2021)](https://dl.acm.org/doi/10.1145/3442188.3445922). All these aspects, which are difficult to control *a priori*, inevitably end up in the data that make up the LMs' pre-training datasets, leading them to language generations that cannot always be considered safe and harmless [(Gehman et al., 2020)](https://aclanthology.org/2020.findings-emnlp.301.pdf).

It is for this reason that efforts in research have been made to try to mitigate these phenomena as much as possible, both from the data point of view and from the point of view of the pre-trained LMs. Among the best known techniques can be found fine-tuning, RLHF [(Bai et al., 2022)](https://arxiv.org/abs/2204.05862) and model steering [(Dathathri et al. 2020)](https://arxiv.org/abs/1912.02164). These techniques turn out to be more than effective in controlling the toxicity in model input/output but, especially in the presence of particularly "tendentious" cases it still remains possible to fool the models that still end up generating potentially toxic or unsafe responses. In addition, the most well-known response pattern to prompts deemed as dangerous is to stop the conversation, trying to stop proceeding to toxic behaviors (e.g., "As an AI Language Model I cannot answer this question, ...").

{% include figure.html path="assets/img/detox_LMs/Example_chatGPT_toxic.png" class="img-fluid rounded z-depth-1" zoomable=true %}
*Toxic Prompt on [ChatGPT](https://openai.com/blog/chatgpt) that generates conversation blocking*

#### Goals

With the following research project, we therefore want to **investigate the detoxification process**, pushing not only the models to be safer but exploring their potential by **allowing them to respond even to potentially toxic prompts** by offering a useful counter narrative to send the conversation forward to reason with the user who authored the original prompt.

As can be guessed, it is imperative that such **a process be as transparent as possible**. For this reason, techniques for interpreting the models themselves will be employed to discover how the models change their generation. This will hopefully lead to discovering not only new features of the models but also what techniques might be most effective for **the safety and effectiveness of the LMs** themselves.


<br/>
## **🔨 Approach**

Of the various techniques previously listed, fine-tuning and reinforcement learning represent the state of the art, also employed by industry for the most modern LMs. The main problem related to the use of these techniques, however, is the size of the models themselves. In fact, over the utlim years, there has been a trend toward growth in the number of parameters in language models, reaching and exceeding hundreds of billions in the case of the largest models (GPT-3/4, Bard, ...). For these reasons, even just performing fine-tuning or applying reinforcement learning techniques seems to be quite impossible on consumer hardware or otherwise accessible to the research community. Even just maintaining a 7B model of parameters, on RAM or VRAM, would take more than 32GB.

#### How to deal with Large LMs?

However, there are several techniques that have emerged over time in the literature that aim to mitigate this type of issue. Indeed, it is possible to load models in Half Precision (16 bits instead of 32 bits) or, even more recently, in 8 bits and 4 bits through quantization techniques [(Dettmers et al., 2022)](https://arxiv.org/abs/2208.07339). These techniques allow dynamic mapping of tensors from the original 32bit model in Full Precision to 16bit tensors and, eventually in 8bit tensors, allowing a theoretical reduction of up to 400% (ideal case without training/inference data, in practice less given the necessary preservation of some parameters).

*Half precision input matrix $$X_{f16} \in \!R^{s×h}$$, can be quantizited as follow*:

$$
X_{i 8} = \biggl \lceil \frac{127 \cdot X_{f16}}{\max{(|{X_{f16}}_{i,j} |)}}  \biggr \rfloor = 
\biggl \lceil \frac{127}{||X_{f16}||_{\infty}} \cdot X_{f16} \biggr \rfloor = 
\lceil {s_x}_{f16} X_{f_16} \rfloor
$$

*Scaling a tensor to his 8-bit version forces the range* $$[-127, +127]$$ *by multiplying with* $$ {s_x}_{f16} $$ *which
is 127 divided by the absolute maximum of the entire tensor. This is equivalent to dividing by the infinity norm and multiplying by 127. More info in the original paper.*

This advantage of matrix representation, however, comes at a cost in the inability to effectively modify the matrices within the model, in other words, to perform weight training. 

#### How to efficiently train quantized Large LMs?

In order to fine-tune or otherwise modify the weights of the model there must be weights in FP32 or FP16 representation. For this very reason, [(J. Hu et al., 2021)](https://arxiv.org/abs/2106.09685) with Low-Rank Adaptation (LoRA) aims to create adapters that, in parallel with the frozen weights of the model, allow one to circumvent the problem by offering trainable lower-rank matrices based on the frozen model. The details of this operation will not be exposed here (for more information look at the paper cited earlier) but it is important to mention how this solution allows not only the training of larger models but is shown to partially succeed in solving the catastrophic forgetting problem as well. The most convenient implementation, being integrated with 🤗 HuggingFace is the one provided by 🤗 [Peft](https://huggingface.co/blog/peft).


#### Letting LMs Respond with Contronarrative

As previously mentioned, the state of the art so far has focused on generic detoxification of LMs, certainly leading them to be less toxic by avoiding responding to compromising prompts or otherwise imposing strong constraints on both the optimization process and the output of the model itself. In fact, based on what has been observed, the same models may be able to articulate more complex responses that capture even the most delicate aspects of the dialogue. Thus, we want precisely to explore this concept further, bringing, through fine-tuning, **the model to a contronarrative generation responsive to the given prompt**. 

For this purpose, [(Bonaldi et al., 2022)](https://aclanthology.org/2022.emnlp-main.549/), a dataset curated by experts is employed to provide accurate answers to prompts regarding topics and/or people particularly susceptible and vulnerable to online hate speech.
The dataset, consists mainly of dialogues (thus with multiple prompt-response pairs); we chose to select each pair while maintaining all its antecedents, exploiting the potential of Chain-of-Thought [(Wei et al., 2022)](https://arxiv.org/abs/2201.11903).


#### Fine-tuning and Reinforcement Learning from (Automatic) Feedback

Fine-tuning and reinforcement learning of the models was employed using the [🥞 RewardLM](https://github.com/DanielSc4/RewardLM) library. The library allows integration of the models with 🤗 HuggingFace (the *de facto* standard for OpenSource model sharing and manipulation), training and monitoring of the results obtained efficiently. In the case of Reinforcement Learning ([RFAF](https://github.com/DanielSc4/RewardLM#-reinforcement-learning-with-automatic-feedback-rlaf)), besides all the hyperparameters involved, it is possible to specify different details of the reward model, being able to choose any classifier or a set more than one of them for greater efficiency.

More information regarding the tools used:
<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-center align-items-center">
    {% include repository/repo.html repository="DanielSc4/RewardLM" %}
</div>
<br/>



##### `Toxicity Meter`: an easy way to measure LMs toxicity

Also provided in the [🥞 RewardLM](https://github.com/DanielSc4/RewardLM) library is a tool for measuring the average toxicity of models, `⚖️ Toxicity Meter`. By default, the tool employs the `RealToxicityPrompts` dataset [(Gehman et al., 2020)](https://aclanthology.org/2020.findings-emnlp.301/). It was therefore possible to quantitatively measure not only the initial toxicity of the different models, but also the post fine-tuning toxicity and RLAF. The toxicity itself can be measured either from any of the model configuration(s) used as reward model for RLAF, or from [Perspective API](https://perspectiveapi.com/), offering a better granularity in the different types of toxicity.


<br/>
## **🔬 Experiment and results**

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse tristique porta nisl, et feugiat nunc dignissim ac. Morbi auctor eget purus at congue. Maecenas iaculis nulla leo, ac vulputate leo accumsan sit amet. Fusce tellus augue, pulvinar imperdiet rutrum quis, aliquet viverra nulla. Donec interdum ex non rhoncus posuere. Sed ullamcorper ex eu egestas eleifend. Cras urna justo, viverra a porttitor quis, sollicitudin vel mauris. 

### Baseline
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse tristique porta nisl, et feugiat nunc dignissim ac. Morbi auctor eget purus at congue. Maecenas iaculis nulla leo, ac vulputate leo accumsan sit amet. 


### Evaluation setup

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse tristique porta nisl, et feugiat nunc dignissim ac. Morbi auctor eget purus at congue. Maecenas iaculis nulla leo, ac vulputate leo accumsan sit amet. 

### Result

| Left aligned | Center aligned | Right aligned |
| :----------- | :------------: | ------------: |
| Left 1       | center 1       | right 1       |
| Left 2       | center 2       | right 2       |
| Left 3       | center 3       | right 3       |


Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse tristique porta nisl, et feugiat nunc dignissim ac. Morbi auctor eget purus at congue. Maecenas iaculis nulla leo, ac vulputate leo accumsan sit amet. 


<br/>
## **🚀 Current status and new research questions**

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse tristique porta nisl, et feugiat nunc dignissim ac. Morbi auctor eget purus at congue. Maecenas iaculis nulla leo, ac vulputate leo accumsan sit amet. Fusce tellus augue, pulvinar imperdiet rutrum quis, aliquet viverra nulla. Donec interdum ex non rhoncus posuere. Sed ullamcorper ex eu egestas eleifend. Cras urna justo, viverra a porttitor quis, sollicitudin vel mauris. 


