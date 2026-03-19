# 04: Embeddings

Statistical NLP improved on rules by learning from data. But the representation it used — word counts and n-gram sequences — had a ceiling.

Two problems remained:

1. Bag of Words treated words as isolated symbols with no relationship to each other. "Dog" and "puppy" were as different as "dog" and "democracy."
2. N-gram models only looked at short windows. Meaning beyond that window was invisible.

Both problems came down to the same root: the model had no representation of what words actually *mean*.

Embeddings were built to solve that.

### What Came Before: One-Hot Encoding

Before embeddings, words were represented as one-hot vectors.

For a vocabulary of just 6 words, one-hot looks like this:

```
          cat   dog   puppy   run   democracy   river
"cat"  → [ 1,    0,     0,    0,       0,        0 ]
"dog"  → [ 0,    1,     0,    0,       0,        0 ]
"puppy"→ [ 0,    0,     1,    0,       0,        0 ]
```

Each vector is mostly zeros with a single 1. This is called **sparse** — most positions are empty. In a real vocabulary of 50,000 words, each vector has 49,999 zeros and one 1.

There is no relationship between any two vectors. "Dog" at position 4,821 and "puppy" at position 7,234 share nothing mathematically. The model has no way to know they are related.

### What an Embedding Is

An embedding replaces the sparse one-hot vector with a **dense** vector — a shorter list where every position has a number, not mostly zeros.

Dense means: no empty slots. Every number in the vector carries information.

But where do those numbers come from? They are not assigned by hand.

The model starts with completely random numbers for every word:

```
           [pos1   pos2   pos3  ...]
"cat"   → [ 0.13,  0.87,  0.42, ...]   ← random, meaningless
"dog"   → [ 0.55,  0.21,  0.78, ...]   ← random, meaningless
"puppy" → [ 0.91,  0.04,  0.33, ...]   ← random, meaningless
```

At this point "dog" and "puppy" still have no relationship — they are just random positions.

Then training adjusts these numbers. The model is given a task: predict which words appear near each other in text. For every sentence in a large corpus, it makes a prediction, checks if it was right, and adjusts all the numbers slightly based on the error.

Because "dog" and "puppy" appear near the same words — "walk", "feed", "bark", "leash" — the model keeps adjusting both of their vectors in the same direction, millions of times.

After training, the numbers look like this:

```
           [pos1   pos2   pos3  ...]
"cat"   → [ 0.72,  0.61,  0.11, ...]   ← learned
"dog"   → [ 0.68,  0.59,  0.14, ...]   ← learned
"puppy" → [ 0.65,  0.62,  0.09, ...]   ← learned
"democracy" → [ 0.10,  0.03,  0.91, ...]   ← learned
```

Cat, dog, and puppy ended up with similar numbers. Democracy did not. The relationship between words is now visible in the representation itself.

### What a Dimension Is

Each position in that vector is called a **dimension**.

A dimension is just an axis — a direction along which something can vary.

- 1 dimension: a number line. One number per point. Distance can only be measured in one direction.
- 2 dimensions: a flat map. Two numbers per point — latitude and longitude. Now you can measure north-south and east-west.
- 3 dimensions: add altitude. Three numbers per point. Now mountains and valleys have different positions.
- 300 dimensions: 300 axes. 300 numbers per point. You cannot visualize it, but the math works exactly the same.

Each dimension in an embedding captures some abstract pattern the model found useful during training. Most do not map onto any human concept — they are directions in a space the model constructed.

What the 300 numbers say together: "here is where this word sits in a 300-axis space shaped by how it is used in text."

### The Core Idea: Distributional Hypothesis

So how does training know where to place each word?

The foundational idea behind embeddings:

**A word is defined by the company it keeps.**

Look at what words naturally fill these blanks:

- "The ___ chased the ball."
- "The ___ barked at the stranger."
- "She took the ___ for a walk."

"Dog", "puppy", and "labrador" all fit. "Democracy" does not.

"Dog" and "puppy" appear near the same words — "walk", "feed", "bark", "leash", "vet" — across millions of sentences. Their contexts overlap heavily.

That overlap is the training signal. Words with overlapping contexts get pulled toward similar positions. "Dog" and "puppy" end up close. "Democracy" ends up far away — because it appears near entirely different words: "election", "freedom", "vote", "government."

The vector is no longer an arbitrary index. It is a position shaped by the word's actual usage in language.

### How Embeddings Are Trained: Word2Vec

Word2Vec (2013) was the technique that showed this could be done efficiently at scale.

The training objective: given a word, predict the words that typically appear around it in a sentence.

For the sentence "The dog chased the ball down the street":

- given "dog", predict: "the", "chased", "the"
- given "chased", predict: "dog", "the", "ball"
- given "ball", predict: "chased", "the", "down"

The model starts with random vectors and adjusts them every time it makes a prediction. No labels are needed. The structure of language itself is the training signal.

### Example: What the Space Shows

After training on a large corpus, the vector space has structure.

Words with similar usage patterns end up near each other:

- "cat" and "dog" end up close → both appear near "pet", "feed", "vet", "walk"
- "doctor" and "nurse" end up close → both appear near "hospital", "patient", "treat"
- "Paris" and "Berlin" end up close → both appear near "capital", "Europe", "city"

The space also supports arithmetic. Because the relationship between word pairs is consistent:

```
"king" − "man" + "woman" ≈ "queen"
"Paris" − "France" + "Italy" ≈ "Rome"
```

The model never learned these rules. They fell out of the patterns in the data.

### How Similarity Is Measured

Closeness in this space is measured by the angle between two vectors — called **cosine similarity**.

Two vectors pointing in nearly the same direction are considered similar. Two vectors pointing in different directions are not — regardless of the actual coordinate values.

This is why the individual numbers in a vector do not mean anything on their own. What matters is the angle between two word vectors, not the value at any single position.

### What This Improved Over Statistical NLP

With Bag of Words, a model trained on sentences about dogs had no information about puppies — they were different symbols.

With embeddings, "dog" and "puppy" have similar vectors. A model can generalize from one to the other. A classifier that learned what negative dog reviews look like can apply that to puppy reviews without ever seeing a puppy review in training.

This was not possible with word counts.

### The Limitation: One Vector Per Word

Classic embeddings assign one fixed vector to each word, regardless of context.

Consider the word `light`:

- "Turn off the light." → a lamp
- "She packed light." → without much weight
- "The color was light blue." → pale shade

All three usages get the same vector — an average of all the contexts where "light" appeared during training. The model cannot tell which meaning applies in a specific sentence.

Embeddings improved representation of meaning across words, but they could not represent which meaning a word carries in a given context.

That problem is called context-sensitivity. Solving it required models that process the full sequence — computing a word's representation based on the words around it in each specific sentence.

### What Came Next

Sequence models were built to process words in order, carrying context forward as they read. That gave each word's representation a chance to be shaped by what came before it.
