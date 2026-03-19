# 07: Encoder-Decoder and Attention

LSTMs could process a sentence and retain information across its full length. That solved the single-sequence problem.

But translation is a different kind of task.

Reading "The cat sat on the mat" and producing "Le chat était assis sur le tapis" is not about processing one sequence — it is about mapping one sequence to another. The source language sentence comes in. A target language sentence comes out. The lengths can differ. The word order can differ. The model needs to understand the full source before it can reliably generate the target.

A single LSTM reading one sequence at a time was not built for this.

---

### The Encoder-Decoder Architecture

The solution was to build two sequence models and connect them.

**The encoder** reads the full source sentence word by word and compresses everything into one vector — its final hidden state.

**The decoder** takes that single vector and generates the target sentence word by word.

```
Source: "The cat sat on the mat."
         ↓
    encoder reads: "The" → "cat" → "sat" → "on" → "the" → "mat"
         ↓
    final hidden state: [ 0.71, -0.23, 0.88, ... ]   ← one vector for the whole sentence
         ↓
    decoder generates: "Le" → "chat" → "était" → "assis" → "sur" → "le" → "tapis"
         ↓
Output: "Le chat était assis sur le tapis."
```

This worked. For short sentences it produced reasonable translations.

---

### The Bottleneck Problem

The architecture had one hard limit: the entire source sentence had to fit into a single fixed-size vector before decoding started.

"The cat sat on the mat" is 6 words. Compressing 6 words into one vector is manageable.

Now use the tourist sentence from the previous chapter:

"The tourist who arrived from Spain, after a long flight through three different airports, finally reached the hotel where the room had been reserved."

This is 28 words. The encoder reads all 28 one by one. Each step updates the hidden state. By the end, the final hidden state is trying to represent:

- who the subject is (the tourist)
- where they came from (Spain)
- how they traveled (long flight, three airports)
- what happened (reached the hotel)
- what state the reservation was in (had been reserved)

All of that in one fixed-size vector.

Words at the beginning — "tourist", "Spain" — were processed first and have been overwritten dozens of times. Their contribution to the final vector is minimal. When the decoder tries to generate the translated version of "the tourist", the information about the tourist may barely exist in the vector it was given.

The longer the sentence, the worse this problem gets.

---

### Attention: Letting the Decoder Look Back

Attention was introduced to break the single-vector bottleneck.

The key change: instead of compressing the source into one vector, keep every encoder hidden state — one per source word.

```
Source: "The   cat   sat   on   the   mat"
         ↓      ↓     ↓    ↓    ↓     ↓
        h_1    h_2   h_3  h_4  h_5   h_6    ← all 6 hidden states kept
```

When the decoder generates each output word, it is allowed to look at all 6 encoder hidden states and decide which source words are most relevant for this specific output word.

**How it computes this:**

1. For the current decoder state, compute a **score** against each encoder hidden state: how relevant is this source word right now?
2. Convert all scores to weights using softmax — they sum to 1
3. Take a weighted sum of all encoder hidden states → this is the **context vector** for this step
4. Use the context vector to predict the next output word

**Example: generating "chat" (French for "cat")**

```
Decoder is generating the second French word.

Attention scores over each source word:
  h_1 ("The")  → score: 0.02
  h_2 ("cat")  → score: 0.91   ← highest
  h_3 ("sat")  → score: 0.03
  h_4 ("on")   → score: 0.02
  h_5 ("the")  → score: 0.01
  h_6 ("mat")  → score: 0.01

Context vector = (0.02 × h_1) + (0.91 × h_2) + (0.03 × h_3) + ...
               ≈ mostly h_2  ← the encoder state for "cat"
```

The decoder focused almost entirely on the encoder state for "cat" when generating "chat." Not because anyone told it to — the attention weights were learned from training.

**What this solved for the tourist sentence:**

With the encoder-decoder bottleneck, the decoder had to generate the full French translation from one vector that barely remembered "tourist."

With attention, when the decoder is generating the French word for "tourist", it can directly attend to `h_1` — the encoder hidden state from when "tourist" was read. No compression. No overwriting. Direct access.

```
Generating French for "tourist":
  h_1 ("The")      → 0.02
  h_2 ("tourist")  → 0.89   ← attends directly to the source word
  h_3 ("who")      → 0.04
  ...
```

The tourist's information does not need to survive 28 steps of hidden state updates. It is available directly.

---

### How Attention Scores Are Computed

The scores are not looked up from a table. They are computed.

At each decoding step, the model has a current decoder hidden state `s`. For each encoder hidden state `h_i`, it computes a score:

```
score(s, h_i) = s · h_i          (dot product — one common approach)
```

The dot product measures how similar the decoder's current state is to each encoder state. A high score means the decoder's current need aligns with what that encoder position encoded.

Those scores are passed through softmax to produce weights that sum to 1. Then the context vector is the weighted sum of all encoder states.

The whole thing is differentiable — gradients flow through it during training. The model learns, through backpropagation, what to attend to for each type of output.

---

### Example: Attention Beyond Translation

Attention is not only useful for translation. It solves any problem where the model needs to connect a current position to a relevant earlier position.

"The trophy did not fit in the suitcase because it was too large."

When processing "it", the model needs to decide: does "it" refer to "trophy" or "suitcase"?

With attention, the model scores every earlier word from the perspective of resolving "it":

```
Attention weights when processing "it":
  "The"       → 0.01
  "trophy"    → 0.78   ← weighted heavily
  "did"       → 0.01
  "not"       → 0.02
  "fit"       → 0.04
  "in"        → 0.01
  "the"       → 0.01
  "suitcase"  → 0.09
  "because"   → 0.03
```

"Trophy" receives the highest weight. The model learned that "large" connects more naturally to something that does not fit (the trophy) than to the container (the suitcase).

Without attention, the model would have to resolve this from its compressed hidden state alone — which may have lost the distinction between the two nouns across that many steps.

---

### Why Attention Mattered Beyond Sequence Models

Attention was added to fix the encoder-decoder bottleneck. But it revealed something more fundamental.

The step-by-step processing of RNNs and LSTMs was not just slow to train — it was architecturally the wrong way to handle long-range relationships. The hidden state was always a compression of the past, never a direct connection to it.

Attention gave any position in a sequence direct access to any other position. No compression. No sequential dependency required.

The question this raised: if attention is what's actually doing the important work — connecting positions to each other — why keep the recurrent structure at all?

What if you built an architecture around attention from the start and removed the step-by-step processing entirely?

That is where the next part begins.
