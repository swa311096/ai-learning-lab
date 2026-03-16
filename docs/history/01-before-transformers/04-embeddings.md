# 04: Embeddings

Statistical models could count words. But counting is not the same as understanding meaning.

Consider the word "dog."

A count-based model treats "dog" and "puppy" as completely separate symbols. It has no way to know they are related. If it has seen many sentences with "dog" but few with "puppy," it will handle them differently — even for tasks where they mean the same thing.

Embeddings changed that.

## The Core Idea

Represent each word as a vector — a list of numbers.

The key insight: words that appear in similar contexts end up with similar vectors.

A model trained on enough text will learn that "dog" and "puppy" tend to appear in similar situations — playing, walking, feeding, training. Their vectors will end up close to each other in numeric space.

## Example: What the Map Shows

Think of each word as a point on a map. Words with similar usage patterns end up near each other.

- "cat" and "dog" end up close → both are household pets
- "doctor" and "nurse" end up close → both appear in medical contexts
- "Paris" and "Berlin" end up close → both appear in sentences about European capitals

You can also do arithmetic. In a trained embedding space:

"king" − "man" + "woman" ≈ "queen"

The model never learned this rule. It fell out of the patterns in the data.

## What This Improved

Embeddings let models generalize across related words.

A model trained on sentences about dogs could now handle sentences about puppies better — because their vectors are close. Word counts could not do that.

## The Limitation

Classic embeddings give each word one fixed vector, regardless of context.

Consider the word `light`:

- "Turn off the light." → a lamp
- "She packed light." → without much weight
- "The color was light blue." → pale shade

All three get the same vector. The model cannot tell them apart.

This is the context problem that embeddings alone could not solve. Fixing it required models that process the full sequence — not just individual words.

## What Came Next

The field moved toward sequence models: architectures that read words in order and can use surrounding context to determine what each word means in each specific sentence.
