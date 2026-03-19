# 02: Rule-Based NLP

If humans understand language through patterns, why not just write those patterns down?

That was the first serious attempt.

### What A Rule-Based System Looks Like

A developer writes explicit logic:

- if the sentence contains these words, classify it as positive
- if the pattern matches this template, extract this entity
- if this grammar rule applies, assign this structure

The system has no ability to learn. It can only do what the rules tell it.

### Why This Made Sense

Language does have structure. Sentences follow grammar. Words carry consistent meanings in specific domains.

If you are extracting dates from a form, or checking spelling against a dictionary, or routing support tickets by keyword — rules work.

The problem is not that rule-based systems never worked. The problem is that they could not scale beyond controlled conditions.

Two examples show this clearly. The first shows a rule-based system that appeared to work — within a narrow range. The second shows what happened when the same approach was applied to a task with more variation.

### Example: ELIZA

ELIZA was a rule-based program built in the 1960s to simulate a conversation.

One of its rules looked something like this:

- if the user says "I feel [X]", respond with "Why do you feel [X]?"

This worked well enough that some users believed they were talking to a real therapist.

- User: "I feel tired."
- ELIZA: "Why do you feel tired?"
- User: "I feel like no one listens to me."
- ELIZA: "Why do you feel like no one listens to you?"

But the system had no understanding. It was pattern matching on surface text. Step outside the patterns and it broke immediately:

- User: "I don't know what I feel."
- ELIZA: no matching rule → generic fallback response

ELIZA was not intelligent. It was a set of templates that looked intelligent under specific conditions.

ELIZA was a conversation system. But the same approach — match a surface pattern, fire a rule — was applied to other types of language tasks too. The result was the same.

### Example: Sentiment Detection

Sentiment detection is the task of reading a piece of text and deciding whether it expresses a positive or negative opinion.

This had a direct business use: product reviews. A company receiving thousands of customer reviews cannot read all of them. If a system could automatically label each one as positive or negative, you could track satisfaction over time, flag complaints, and prioritize responses.

Humans do this easily. We read a review and know within a sentence whether the customer is happy or not. So the assumption was: if we can identify the words that signal positive or negative, we can write that down as rules.

A rule-based sentiment system might look like:

- if the text contains "good", mark it positive
- if the text contains "bad", mark it negative

This works for:

- "The movie was good."
- "The movie was bad."

But it breaks on:

- "The movie was not bad." → negative word, positive meaning
- "The movie looked good, but it was boring." → mixed signal
- "Good luck finishing that disaster." → sarcasm

Every new case requires a new rule. Every new rule risks conflicting with an existing one.

The problem was not the volume of reviews. The problem was the variation in how people write. No finite set of rules could cover it.

### What Rule-Based Systems Could Do

In narrow, controlled domains:

- spell checking and grammar correction
- extracting structured fields from forms and templates
- keyword-based routing and classification
- parsing in domains with limited vocabulary

### What They Could Not Do

Anything that required handling variation, context, or language outside the cases the rules were designed for.

This created three compounding problems:

1. the system becomes hard to maintain as rules multiply
2. rules conflict with one another
3. performance breaks as soon as input deviates from what was anticipated

### What This Taught the Field

Writing rules by hand does not scale.

Language is too varied. Edge cases are not exceptions — they are the norm. Every new domain, every new user population, every new phrasing requires new rules.

The lesson was not that language is too hard to process. The lesson was that the approach of encoding rules manually was the wrong foundation.

### What Came Next

This pushed researchers toward a different question:

instead of telling the system how language works, can the machine learn those patterns from data?

That is where statistical NLP begins.
