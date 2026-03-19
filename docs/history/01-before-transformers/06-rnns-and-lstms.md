# 06: RNNs and LSTMs

Embeddings gave each word a position in a meaningful space. But every word was still processed independently — one embedding per word, no connection between them.

The problem this left open: meaning often depends on what came before.

- "not good" — BoW treated these as two independent words and got the sentiment wrong
- "The cat sat on the mat because **it** was tired" — which word does "it" refer to?
- "The bank near the **river**" — means something different than "the bank near the office"

N-grams handled local order but only within a short window. Embeddings captured word meaning but not word relationships within a sentence.

What the field needed: a model that could read a sentence from left to right and carry information forward as it went.

That is what RNNs were built to do.

---

## Why RNNs Came

Before RNNs, the best available approaches were:

- n-gram models → look at last 2-3 words only
- embeddings → each word represented independently, no sentence-level context

Neither could handle dependencies that spanned more than a few words. Neither had a way to build up a representation of a full sentence.

The insight behind RNNs: process the sentence one word at a time. At each step, combine the current word with everything seen so far. Pass that combination forward.

---

## What "Recurrent" Means

Sequence models are called **recurrent** because they loop — the output of one step feeds back as input to the next.

At each step, the model takes:

1. the current word's embedding (x)
2. the hidden state from the previous step (h)

And produces a new hidden state.

![RNN unrolled over time — each cell takes a word input and the previous hidden state](assets/rnn-unrolled.png)

Reading the diagram left to right:

- `x0, x1, x2, x3` — word embeddings fed in at each step
- `h` arrows flowing right — hidden state passed forward
- `h0, h1, h2, h3` — hidden state output at each step

The hidden state is a fixed-size vector — a compressed summary of everything read so far.

For "The cat sat on the mat":

```
Step 1:  embed("The")  + h_start → h_1
Step 2:  embed("cat")  + h_1     → h_2  ← carries "The cat"
Step 3:  embed("sat")  + h_2     → h_3  ← carries "The cat sat"
Step 4:  embed("on")   + h_3     → h_4
Step 5:  embed("the")  + h_4     → h_5
Step 6:  embed("mat")  + h_5     → h_6  ← carries full sentence
```

Each step updates the hidden state. Step 3 cannot run until step 2 is done.

---

## What RNNs Solved

**Negation in sentiment:**

BoW for "The movie was not bad":
```
vector: {movie:1, not:1, bad:1}
"bad" scores negative → predicted: negative  ← wrong
```

RNN for "The movie was not bad":
```
step 4: reads "not"  → h_4 carries negation signal
step 5: reads "bad"  → h_5 combines negation + negative word
```

Through training, the model learns this sequence pattern — negation followed by negative word — appears in positive reviews. Order is preserved. BoW had no way to encode it.

**Co-reference resolution:**

"The cat sat on the mat because it was tired."

An RNN reaches "it" with a hidden state updated by every word before it, including "cat" and "mat." Through training, the model learns to track the subject and use it to resolve "it" correctly.

**Language modeling beyond short windows:**

N-grams with N=3 use only the last 2 words as context. An RNN at any step has a hidden state built from the entire sequence — not just the last few words.

---

## What RNNs Could Not Do

Two problems limited RNNs.

**The hidden state bottleneck:**

The hidden state is fixed-size regardless of how long the input is. For a 3-word sentence, fine. For a 40-word sentence, the model compresses 40 words into the same fixed vector.

Early information gets overwritten:

"The tourist who arrived from Spain, after a long flight through three different airports, finally reached the hotel where the ___ had been reserved."

By the time the model reaches "___", the hidden state has been updated 15 times since "tourist." The subject — the word needed to complete the sentence — may be largely gone.

**Vanishing gradients:**

Training requires backpropagating through every step of the sequence. As covered in ch05, gradients shrink as they travel backward. In a 40-step sequence, the gradient for step 1 travels through 40 backpropagation steps and fades to near zero. The model cannot learn that something at step 1 is connected to something at step 40.

---

## LSTMs: Solving the Forgetting Problem

Take the tourist sentence that broke the RNN. With a basic RNN, by the time the model reaches "___", "tourist" has largely faded from the hidden state.

The **LSTM** (Long Short-Term Memory) was built to fix this.

Instead of one hidden state that gets fully rewritten at every step, LSTMs maintain two separate pieces of memory:

- **hidden state** — short-term, what to expose right now
- **cell state** — long-term, carried forward with only controlled changes

The cell state runs like a conveyor belt through the sequence. Information can be written onto it or erased from it — but only through learned gates. It does not get fully overwritten at each step.

LSTMs add three gates:

| Gate | Question it answers | Effect on the tourist sentence |
|---|---|---|
| Forget gate | What should I erase from long-term memory? | Erases "Spain", "three airports" — keeps "tourist = subject" |
| Input gate | What new information should I write to long-term memory? | Adds "reached hotel" — context now: "tourist reached hotel" |
| Output gate | What part of memory should I expose right now? | Exposes "tourist" as subject when predicting what was reserved |

![RNN vs LSTM on the tourist sentence — RNN hidden state fades, LSTM cell state retains tourist](assets/rnn-vs-lstm-tourist.png)

Left panel: RNN hidden state bar fades to near-white by "hotel ___" — "tourist" is gone. Right panel: LSTM cell state bar stays uniformly dark — "tourist" is retained through every step.

**What changes across the sentence:**

```
Read "The tourist":
  → cell state writes: "tourist = subject"
  → hidden state: short-term signal about "tourist"

Read "who arrived from Spain":
  → forget gate: "Spain" not critical → partially erased
  → cell state still holds: "tourist = subject"

Read "after a long flight through three different airports":
  → forget gate: travel details not needed → erased
  → cell state still holds: "tourist = subject"

Read "finally reached the hotel where the ___ had been reserved":
  → output gate exposes: "tourist = subject"
  → model predicts: "room" (reserved for the tourist)
```

A basic RNN at this point would have a hidden state dominated by "hotel", "had been" — with "tourist" largely gone. The LSTM cell state kept it.

---

## How Does the Gate Know What to Keep?

The gate table says "erase Spain, keep tourist" — that sounds like pre-programmed knowledge. It is not.

Each gate is a small neural network. At every step it takes:

1. the current word's embedding
2. the previous hidden state

It runs them through a linear layer and a **sigmoid function**, producing a value between 0 and 1 for every dimension of the cell state:

```
forget gate output = sigmoid( W_f × [current_word, prev_hidden] + b_f )
```

- `0.0` → erase this dimension completely
- `1.0` → keep this dimension completely
- `0.3` → keep 30%, erase 70%

That output vector is multiplied element-wise with the cell state. Dimensions where the gate output is near 0 get erased. Near 1, they survive.

**How does it learn which dimensions to protect?**

Through backpropagation. The gate weights start random and adjust over training. Over thousands of sentences, the model learns:

- subject information ("tourist", "she", "the company") is almost always needed later — for verb agreement, co-reference, predicate completion
- prepositional details ("from Spain", "via three airports") rarely affect the sentence's end

After training, when reading "from Spain", the forget gate might produce:

```
cell dimension encoding "current subject" → gate output: 0.94  (keep)
cell dimension encoding "recent location" → gate output: 0.11  (erase)
cell dimension encoding "verb tense"      → gate output: 0.87  (keep)
```

The gate did not know Spain was unimportant. It learned that words following "arrived from" rarely affect the subject slot at the end of a sentence.

**GRUs** (Gated Recurrent Units) simplified this into two gates instead of three with comparable results and less computation.

LSTMs and GRUs became the standard for sequence tasks through the 2010s.

---

## What Came Next

LSTMs solved the forgetting problem within a single sequence. A model could now read a sentence and retain what mattered across its full length.

But one major task remained out of reach: mapping a sequence in one language to a sequence in another.

Translation required reading an entire source sentence, building a representation of it, and then generating a new sentence in a different language. That called for a different architecture built on top of sequence models.

That is where the next chapter begins.
