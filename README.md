# Pivotal NLP papers

This section traces the progression from advanced Recurrent Neural Network (RNN) models with attention to the complete dominance of the Transformer architecture, which laid the groundwork for modern Large Language Models (LLMs).

* [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025) (Luong, et al., 2015) - A foundational paper that explores and simplifies different attention mechanisms for sequence-to-sequence models, providing a clear starting point for implementing attention on top of RNNs.
[(Code Repo)](https://github.com/lmthang/nmt.hybrid)

* Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation (Wu, et al., 2016) - Showcases the power of a deep LSTM network with attention at a massive scale, representing the pinnacle of the pre-Transformer era of machine translation.

* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) (Vaswani, et al., 2017) - The revolutionary paper that introduced the Transformer architecture, abandoning recurrence entirely in favor of self-attention. This is the essential starting point for understanding modern LLMs [(Code Repo)](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)

* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805) (Devlin, et al., 2018) - Introduced the concept of pre-training a deep bidirectional Transformer using a Masked Language Model (MLM) objective, fundamentally changing the paradigm for transfer learning in NLP.

* Language Models are Unsupervised Multitask Learners (GPT-2) (Radford, et al., 2019) - Demonstrated that a large, autoregressive Transformer model trained on a massive and diverse dataset could perform a wide range of NLP tasks in a zero-shot setting, without any task-specific training.

* RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu, et al., 2019) - A replication study of BERT that showed its performance was significantly improved by training longer, on more data, and with careful hyperparameter tuning, establishing a stronger baseline for pre-trained models.

* Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5) (Raffel, et al., 2020) - Proposed a unified framework that treats every NLP task as a text-to-text problem, allowing a single model to be used for a wide variety of tasks from translation to classification by using task-specific prefixes.

* Language Models are Few-Shot Learners (GPT-3) (Brown, et al., 2020) - Scaled the Transformer architecture to 175 billion parameters, demonstrating that extremely large models can achieve strong performance on many tasks with only a few examples (few-shot) or even no examples (zero-shot) provided in the prompt.

* BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (Lewis, et al., 2020) - Presented a pre-training scheme for sequence-to-sequence models that combines a bidirectional encoder (like BERT) with an autoregressive decoder (like GPT), making it particularly effective for generative tasks.

* ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (Clark, et al., 2020) - Introduced a more sample-efficient pre-training task called replaced token detection (RTD), where a model learns to distinguish real input tokens from plausible fake ones, leading to faster training and better downstream performance.
