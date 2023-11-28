---
layout: post
title: "In-depth Notes to understand more about Transformers"
date: 2023-09-10 11:59:00-0400
description: Start of the truly in-depth study of the state-of-the-art for Mechanistic Interpretability
categories: study
giscus_comments: false
related_posts: false
toc:
  beginning: false
---

## Why this post?
With the start of my PhD, I want to focus more and more on the **interpretability** of models, in as much detail as possible. Inspired by the various works in the literature, in particular that which has been carried out by Anthropic (e.g. [here](https://transformer-circuits.pub/2021/framework/index.html) and [here](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)), Google DeepMind and a number of other researchers around the world, I would like to be able to **contribute scientific output on this topic**. 
I also hope that the material and sources here may be useful for someone else who, at the same time or in the near future, decides to explore this field with me.

#### What will be the first steps?
Surely there are far more experienced people around on these topics. My knowledge for now is limited to basic notions of linear algebra, mathematics and practical applications of transformer networks in the past, certainly useful for 'basic' research but equally expandable for really significant achievements. 

In the beginning, particular attention will be paid to going into detail on each individual topic, in no particular order but being guided by pure curiosity in understanding new things that I may have taken for granted until recently. Gradually the point will come where I cannot go any lower than this with my study and only at that point will I be able to concentrate on the real research, continuing with the same curiosity but in directions not yet explored.

#### Why am I writing all this here?
Because studying is by no means easy, commitments are many (even outside the PhD) but I still want to achieve goals. This blog post exposes me and therefore forces me to stick to a timeline. And then a plus I like to write random things 😅.

#### What will I post?
Everything I produce over time. It could be handwritten notes on a specific topic, written code with experiments or exercises to apply the theory, or resources that I consider fundamental if they have not already been mentioned previously in the material produced.

Obviously I will continue to update this post as I do things, move forward and proceed in the direction set.

---

# **Chapter 1: Transformers, of course**

Mainly relying on random resources on transformers online ([here](https://transformer-circuits.pub/2021/framework/index.html) and [here](https://deepgram.com/learn/capturing-attention-decoding-the-success-of-transformer-models-in-natural-language-processing)), I started to take notes on how they work. I will not focus too much on the whole architecture as a whole but have mainly tried to focus on the attention mechanism and MLP and how these components act on the [residual stream](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=DHp9vZ0h9lA9OCrzG2Y3rrzH) of a transformer. I will therefore leave out for now topics such as embeddings and other trying to implement this using [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) ([here](https://einops.rocks/1-einops-basics/) a Python tutorial) for the various matrix operations, which I knew existed but had never used in practice. Below are handwritten notes to that effect.

<div style="width: 100%; max-width: 100%;">
    <object data="https://www.danielsc4.it/assets/pdf/interpretability_study/Transformer notes.pdf" type="application/pdf" style="width: 100%; height: 500px;">
        <embed src="https://www.danielsc4.it/assets/pdf/interpretability_study/Transformer notes.pdf">
            <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://www.danielsc4.it/assets/pdf/interpretability_study/Transformer notes.pdf">Download PDF</a>.</p>
        </embed>
    </object>
</div>





