# 05: Sequence Models and Attention

Embeddings gave models a way to represent word meaning. But they still processed each word independently.

What happens when the meaning of a word depends on what came earlier in the sentence?

## Example: Why Sequence Matters

Consider:

"The cat sat on the mat because it was tired."

What does "it" refer to? The cat. Not the mat.

A model that reads words in isolation has no way to connect "it" to "cat." It needs to carry information forward across the sentence.

That is what sequence models were built to do.

## Sequence Models

This is where RNNs, LSTMs, and GRUs became important.

These models process text one word at a time and maintain a running hidden state — a compressed summary of what came before. Each new word updates that state.

## What They Improved

Compared to count-based and embedding-only methods, sequence models handled:

- translation
- text generation
- speech recognition
- sequence labeling tasks

## The Main Limitation

Sequence models had two problems.

The first was the hidden state. It is a fixed-size vector, regardless of how long the sequence is.

For short sentences, this works. For longer ones, early information gets compressed and overwritten.

Consider:

"The tourist who arrived from Spain, after a long flight through three different airports, finally reached the hotel where the ___ had been reserved."

By the time the model reaches "___", useful information about "tourist" may have been lost. The model struggles to maintain a reliable connection across that distance.

The second problem was training speed. Because these models process one word at a time in order, they cannot be parallelized easily. Training on large amounts of text was slow.

Both problems needed solving before language models could scale.

## Attention

Attention was introduced to address this.

Instead of depending only on the running hidden state, the model gets to look back at every earlier word and decide which ones are relevant right now.

## Example: Attention in Action

In the sentence:

"The trophy did not fit in the suitcase because it was too large."

When the model reaches "it", it needs to decide whether "it" refers to "trophy" or "suitcase."

With attention, the model can look back at both words and weight them. Because "large" connects more naturally to something that doesn't fit (the trophy, not the container), attention can learn to point "it" toward "trophy."

A model relying only on a compressed hidden state is much more likely to get this wrong.

## Why Attention Mattered Beyond Sequence Models

Attention did not just patch sequence models.

It pointed toward a new idea: what if you removed the step-by-step processing entirely and built an architecture around attention from the start?

That is where the next part begins.
