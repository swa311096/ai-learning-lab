# 02: Transformers and Scaling

This section covers the transition from attention-based models to modern LLMs.

#### Stage 1: The Transformer Shift

#### The Breakthrough

Transformers replaced step-by-step sequence processing with an architecture built around attention.

The headline idea was:

tokens can directly attend to other relevant tokens instead of relying only on a running hidden state.

#### Why It Mattered

This improved two things at once:

- better handling of long-range relationships
- much better parallelization during training

That combination made large-scale training more practical.

#### Simple Example

In a sentence like:

"The trophy does not fit in the suitcase because it is too large."

the word "it" should connect to "trophy", not "suitcase".

Attention helps the model connect those related words more directly.

#### Stage 2: Pretraining

#### The Shift

Instead of training a separate model for every task, researchers started training very large models on huge amounts of text.

One common objective was:

predict the next token.

#### Why It Mattered

A single model could learn broad language patterns and then be adapted to many tasks.

This is the basic foundation of modern GPT-style systems.

#### Simple Example

If a model sees enough text, it learns patterns such as:

- how arguments are structured
- how explanations are usually written
- how code tends to look

It does not "understand" in the human sense, but it becomes very good at predicting plausible continuations.

#### Stage 3: Scaling

#### The Observation

As models, datasets, and compute grew, capability often improved.

This led to the scaling era.

#### Why It Mattered

People realized that large enough models could perform a surprisingly wide range of tasks with the same general training setup.

#### Terms That Became Common

- parameters
- scaling laws
- context window
- inference cost

#### Simple Example

A larger model may be better at:

- summarizing a messy document
- following nuanced instructions
- writing more coherent code

But it also becomes more expensive to run.

#### Stage 4: Context Windows

#### What It Means

The context window is how much text the model can process in one request.

#### Why It Mattered

As models became useful for more complex tasks, people wanted them to see:

- longer documents
- longer chat histories
- more examples
- more state for multi-step tasks

#### Practical Example

If a model can only see a short amount of text, it may forget earlier requirements in a conversation.

If it can see much more text, it can keep more of that state in one request.

#### Important Caveat

A bigger context window is helpful, but it does not automatically mean better reasoning or better memory quality.

#### Stage 5: Mixture-of-Experts and Efficiency

#### The Problem

Very large dense models are expensive to run.

#### The Approach

Mixture-of-Experts, or MoE, activates only part of the model for a given token or input instead of using the full network every time.

#### Why It Mattered

This can improve efficiency while still keeping the overall model very large.

#### Simple Example

Think of it like a company with many specialists. Not every specialist works on every problem. Only a relevant subset gets involved for each task.

#### What To Remember From This Section

- transformers made large-scale language modeling much more practical
- pretraining created general-purpose language models
- scaling increased capability and cost together
- context windows became a major product and engineering concern
- MoE is one example of efficiency-driven architecture design

The next section explains how these models became useful assistants and AI systems.
