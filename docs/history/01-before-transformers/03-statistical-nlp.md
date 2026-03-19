# 03: Statistical NLP

The previous chapter ended with a question: instead of telling the system how language works, can the machine learn those patterns from data?

Statistical NLP was the answer to that question.

### What "Statistical" Actually Means

In statistical NLP, learning from data means counting.

You take a large corpus of text and count:

- how often words appear together
- how often a word follows another word
- how often a word appears in positive vs negative examples

No rules are written by hand. The counts derived from the data become the model.

This was a fundamental change in approach. The system does not need an expert to encode patterns. It needs data.

### How Text Was Represented: Bag of Words

Before the model can count anything, it needs a way to represent text as numbers.

The approach used in most of this era: Bag of Words.

Take a sentence and convert it into a count of each word, ignoring order:

- "The movie was good." → `{the: 1, movie: 1, was: 1, good: 1}`
- "The movie was not good." → `{the: 1, movie: 1, was: 1, not: 1, good: 1}`

The representation is a vector of word counts. That vector is what the model operates on.

Inference works by combining the signal each word carries from training. If "good" appeared in 78% of positive reviews during training, then seeing "good" in the vector pushes the prediction toward positive.

But here is the problem: "not good" produces a vector that still contains "good". The model sees the positive signal from "good" and may still classify the sentence as positive — because it has no way to know that "not" reversed its meaning. The words are counted individually. Their relationship to each other is gone.

This is the same failure as the rule-based approach, at a different level. The rule-based system did not handle negation because it had no rule for it. The statistical system does not handle negation because its representation throws away word order entirely.

Beyond negation, order matters in a broader sense too. "The dog bit the man" and "The man bit the dog" produce nearly identical vectors. The model cannot distinguish them.

### Example: Naive Bayes for Sentiment

Chapter 02 showed rule-based sentiment detection failing. Statistical NLP gave a better answer to the same problem.

Imagine you have 1000 labeled product reviews: 500 marked positive, 500 marked negative.

Training is just counting:

- "good" appears in 78% of positive reviews and 12% of negative ones → strong positive signal
- "terrible" appears in 4% of positive reviews and 71% of negative ones → strong negative signal
- "the" appears in 97% of both → no signal

When a new review arrives, the model scores it against both labels by multiplying the probabilities of each word:

New review: "not good"

- P(positive) = P("not" | positive) × P("good" | positive) = 0.30 × 0.80 = 0.24
- P(negative) = P("not" | negative) × P("good" | negative) = 0.40 × 0.10 = 0.04

The model picks the higher score → predicts **positive**.

But "not good" is negative. The model got it wrong — because "good" carried a strong positive weight and "not" could not cancel it. Each word is scored independently. Their relationship to each other does not exist in this model.

Now try an unambiguous case: "absolutely terrible"

- P(positive) = P("absolutely" | positive) × P("terrible" | positive) = 0.10 × 0.05 = 0.005
- P(negative) = P("absolutely" | negative) × P("terrible" | negative) = 0.25 × 0.75 = 0.1875

The model picks negative → correct.

When the words in a sentence are independently strong signals, Naive Bayes works. When meaning depends on how words interact — negation, sarcasm, qualification — it breaks down for the same reason Bag of Words does: order and relationship between words are not represented.

No rule was written. The data decided what each word means for this task. That was the real advance over rule-based systems. But the representation it operated on — a bag of independent words — was still the ceiling.

### Example: N-Grams for Language Prediction

N-grams are sequences of N consecutive words.

- N=1 → unigram: a single word → `["president"]`
- N=2 → bigram: two words → `["the president"]`, `["president of"]`, `["of the"]`
- N=3 → trigram: three words → `["the president of"]`, `["president of the"]`

An n-gram model scans a large text corpus and builds a frequency table: how often does each sequence appear, and what word tends to follow it?

For a bigram model, the question is always: **given the last word, what word comes next?**

From a large corpus, the table for "United" might look like:

| Sequence | Count in corpus | Probability |
|---|---|---|
| United → States | 48,210 | 0.61 |
| United → Kingdom | 21,030 | 0.27 |
| United → Nations | 6,400 | 0.08 |
| United → Desk | 3 | 0.00 |

Given "United ___", the model picks "States" — not because it understands geography, but because that transition appeared most often in training.

**Why not always use a larger N?**

A trigram model looks at the last two words instead of one, which gives more context:

- bigram: given "United", predict next word
- trigram: given "of the United", predict next word

The trigram has more context and can make a better prediction. But there is a cost: the larger N is, the more specific the sequence, and the fewer times you have seen that exact sequence in training. A 5-gram sequence might appear only once or not at all — leaving the model with no basis for a prediction.

This is the core tension in n-gram models: larger N = more context but less data to estimate from.

And no matter what N you choose, the model only looks at the last N-1 words. Everything before that is invisible. A sentence that depends on a word from 20 words ago is completely outside the window.

### What Statistical NLP Could Do

- text classification: sentiment, spam detection, topic labeling
- language modeling over short sequences
- part-of-speech tagging and named entity recognition
- components of early machine translation

### What It Could Not Do

The two approaches had different limitations.

**Bag of Words lost order entirely.**

Bag of Words was used for classification tasks — sentiment, spam, topic labeling. In classification, you are not predicting the next word. You are labeling the whole document.

For that, Bag of Words converted each document into a count of words with no regard for sequence. "The dog bit the man" and "The man bit the dog" produce the same vector. The model cannot tell them apart.

This is not a problem for n-grams — n-gram models do use order. "United States" and "States United" are different bigrams and are tracked separately. But Bag of Words, which powered the classification side of statistical NLP, discarded order completely.

**N-gram models could not handle long-range context.**

N-gram models preserved local order but only within their window. Consider:

"The cat that sat on the mat, which had been there since morning, finally ___ up."

By the time the model reaches the blank, "cat" is 15 words back — far outside any practical window. The model has no access to it. It predicts based on the words immediately around the blank, which may produce something grammatically plausible but wrong.

The larger the gap between two related words, the less the model could use that relationship.

### What This Proved

Statistical NLP demonstrated that language systems improve when they learn from data rather than depending on rules written by hand.

That was a shift in the foundation of the field.

The limitation it left open: counting words — even with probabilities — does not capture what words mean or how meaning shifts with context.

### What Came Next

The next question was about representation.

Instead of representing words as counts, could the model learn something about what each word actually means?

That leads to embeddings.
