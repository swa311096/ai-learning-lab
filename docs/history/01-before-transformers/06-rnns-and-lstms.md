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

### Why RNNs Came

Before RNNs, the best available approaches were:

- n-gram models → look at last 2-3 words only
- embeddings → each word represented independently, no sentence-level context

Neither could handle dependencies that spanned more than a few words. Neither had a way to build up a representation of a full sentence.

The insight behind RNNs: process the sentence one word at a time. At each step, combine the current word with everything seen so far. Pass that combination forward.

---

### What "Recurrent" Means

Sequence models are called **recurrent** because they loop — the output of one step feeds back as input to the next.

At each step, the model takes:

1. the current word's embedding (x)
2. the hidden state from the previous step (h)

And produces a new hidden state.

RNN unrolled over time — each cell takes a word input and the previous hidden state

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

### What RNNs Solved

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

### What RNNs Could Not Do

Two problems limited RNNs.

**The hidden state bottleneck:**

The hidden state is fixed-size regardless of how long the input is. For a 3-word sentence, fine. For a 40-word sentence, the model compresses 40 words into the same fixed vector.

Early information gets overwritten:

"The tourist who arrived from Spain, after a long flight through three different airports, finally reached the hotel where the ___ had been reserved."

By the time the model reaches "___", the hidden state has been updated 15 times since "tourist." The subject — the word needed to complete the sentence — may be largely gone.

**Vanishing gradients:**

Training requires backpropagating through every step of the sequence. As covered in ch05, gradients shrink as they travel backward. In a 40-step sequence, the gradient for step 1 travels through 40 backpropagation steps and fades to near zero. The model cannot learn that something at step 1 is connected to something at step 40.

---

### LSTMs: Solving the Forgetting Problem

Take the tourist sentence that broke the RNN. With a basic RNN, by the time the model reaches "___", "tourist" has largely faded from the hidden state.

The **LSTM** (Long Short-Term Memory) was built to fix this.

Instead of one hidden state that gets fully rewritten at every step, LSTMs maintain two separate pieces of memory:

- **hidden state** — short-term, what to expose right now
- **cell state** — long-term, carried forward with only controlled changes

The cell state runs like a conveyor belt through the sequence. Information can be written onto it or erased from it — but only through learned gates. It does not get fully overwritten at each step.

LSTMs add three gates:


| Gate        | Question it answers                                      | Effect on the tourist sentence                                 |
| ----------- | -------------------------------------------------------- | -------------------------------------------------------------- |
| Forget gate | What should I erase from long-term memory?               | Erases "Spain", "three airports" — keeps "tourist = subject"   |
| Input gate  | What new information should I write to long-term memory? | Adds "reached hotel" — context now: "tourist reached hotel"    |
| Output gate | What part of memory should I expose right now?           | Exposes "tourist" as subject when predicting what was reserved |


RNN vs LSTM on the tourist sentence — RNN hidden state fades, LSTM cell state retains tourist

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

### How Does the Gate Know What to Keep?

The table above says "erase Spain, keep tourist" — but who decided that? No one programmed that rule. Here is how it actually works, using the tourist sentence step by step.

The sentence being processed:

```
"The tourist who arrived from Spain, after a long flight through three different airports,
 finally reached the hotel where the ___ had been reserved."
```

**Step 1: The cell state is a list of numbers**

The cell state is a vector — say, 256 numbers. Think of it as 256 slots.

After the LSTM reads the first two words — "The tourist" — the cell state might look like:

```
Current word: "tourist"
Previous hidden state: carries signal from "The"

Cell state after this step:
  slot 0:   0.82   ← something about "subject of the sentence"
  slot 1:  -0.11   ← something about verb tense
  slot 2:   0.67   ← something about noun type
  ...
  slot 255: 0.03
```

No one labeled these slots. The model figured out through training that it was useful to store "subject" information in slot 0. The slots have no names — they just hold numbers, and the model learned to use them in a way that helps it predict correctly.

**Step 2: The forget gate produces one number per slot — grounded in the current word**

The sentence continues. Now the model is at:

```
Words read so far:  "The tourist who arrived from ___"
Current word:       "Spain"
Previous hidden state: carries compressed signal of "The tourist who arrived from"
```

The forget gate takes these two inputs — `embed("Spain")` and the previous hidden state — and produces one number between 0 and 1 for each of the 256 slots:

```
forget gate output at the word "Spain":
  slot 0:  0.94   ← keep  (this is where "tourist = subject" lives)
  slot 1:  0.87   ← keep  (verb tense still relevant)
  slot 2:  0.11   ← erase (location detail — not needed for predicting the end)
  ...
```

The gate is not reading "Spain" and thinking "this is a country, countries are not important." It is reading the embedding of "Spain" combined with the hidden state's signal of "arrived from" — and that combined pattern, after training, produces low values for the location slot.

**Step 3: The cell state is updated slot by slot**

The forget gate output is multiplied with the cell state, one slot at a time:

```
new cell state = forget gate output × old cell state

slot 0:  0.94 × 0.82  =  0.77   ← tourist signal mostly kept
slot 1:  0.87 × (-0.11) = -0.10  ← verb tense mostly kept
slot 2:  0.11 × 0.67  =  0.07   ← location detail nearly erased
```

Slots where gate output is near 1 → cell value survives.
Slots where gate output is near 0 → cell value is wiped.

This repeats at every word. When the model reads "after a long flight through three different airports", the forget gate again produces near-0 for the location/detail slots and near-1 for the subject slot. By the time the model reaches "the hotel where the ___", slot 0 still holds 0.70-something — "tourist" has survived.

The gate did not know Spain was unimportant. It learned — through backpropagation across thousands of sentences — that words following "arrived from" rarely change what is needed at the end of a sentence.

**GRUs** (Gated Recurrent Units) simplified this into two gates instead of three with comparable results and less computation.

LSTMs and GRUs became the standard for sequence tasks through the 2010s.

---

### What Came Next

LSTMs solved the forgetting problem within a single sequence. A model could now read a sentence and retain what mattered across its full length.

But one major task remained out of reach: mapping a sequence in one language to a sequence in another.

Consider:

```
Input:  "The cat sat on the mat."
Output: "Le chat était assis sur le tapis."
```

An LSTM can read the English sentence and build up a hidden state. But then what? The decoder needs to generate French — word by word — from whatever the encoder left behind. The entire meaning of the English sentence has to be compressed into one fixed-size vector before a single French word is produced.

For 6 words that is manageable. For the tourist sentence — 28 words, with details spread across the whole sentence — the compressed vector loses too much. Words processed early barely survive to the end.

Translation required a different architecture built on top of sequence models, one that did not force the entire source sentence through a single bottleneck.

That is where the next chapter begins.