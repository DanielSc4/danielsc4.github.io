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


# **Chapter 1: Transformers, of course**

Mainly relying on random resources on transformers online ([here](https://transformer-circuits.pub/2021/framework/index.html) and [here](https://deepgram.com/learn/capturing-attention-decoding-the-success-of-transformer-models-in-natural-language-processing)), I started to take notes on how they work. I will not focus too much on the whole architecture as a whole but have mainly tried to focus on the attention mechanism and MLP and how these components act on the [residual stream](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=DHp9vZ0h9lA9OCrzG2Y3rrzH) of a transformer. I will therefore leave out for now topics such as embeddings and other trying to implement this using [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) ([here](https://einops.rocks/1-einops-basics/) a Python tutorial) for the various matrix operations, which I knew existed but had never used in practice. Below are handwritten notes to that effect.

<div style="width: 100%; max-width: 100%;">
    <object data="https://www.danielsc4.it/assets/pdf/interpretability_study/Transformer notes.pdf" type="application/pdf" style="width: 100%; height: 500px;">
        <embed src="https://www.danielsc4.it/assets/pdf/interpretability_study/Transformer notes.pdf">
            <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://www.danielsc4.it/assets/pdf/interpretability_study/Transformer notes.pdf">Download PDF</a>.</p>
        </embed>
    </object>
</div>





