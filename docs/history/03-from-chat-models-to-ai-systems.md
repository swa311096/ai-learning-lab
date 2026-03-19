# 03: From Chat Models to AI Systems

This section covers what happened after large pretrained models became useful.

The short version:

the field moved from "can a model generate good text?" to "how do we make systems around the model useful, reliable, and controllable?"

#### Stage 1: Instruction Tuning

#### The Problem

A pretrained model can generate fluent text, but that does not mean it behaves like a helpful assistant.

#### The Shift

Instruction tuning trains the model on examples of user requests and desired responses.

#### Why It Mattered

This made models much better at following requests like:

- summarize this
- extract the key facts
- answer in bullet points
- return JSON

#### Simple Example

A plain pretrained model might continue a piece of text in a generic way.

An instruction-tuned model is more likely to interpret:

"Summarize this email in 3 bullets"

as a real task with a useful output format.

#### Stage 2: Alignment and RLHF

#### The Problem

Even instruction-tuned models can still be unhelpful, unsafe, or badly aligned with human preferences.

#### The Shift

One major approach was RLHF:

Reinforcement Learning from Human Feedback.

The system uses human preference signals to push the model toward more helpful behavior.

#### Why It Mattered

This improved:

- helpfulness
- formatting
- safety behavior
- conversational usability

#### Important Caveat

Alignment does not make the model automatically factual.

It can still sound confident and still be wrong.

#### Stage 3: RAG

#### The Problem

Models are not reliable long-term knowledge stores, especially for fresh, private, or domain-specific information.

#### The Shift

Systems started retrieving relevant information from outside the model and giving that information to the model before asking for an answer.

This is RAG:

Retrieval-Augmented Generation.

#### Why It Mattered

RAG helps with:

- fresh information
- internal documents
- traceable sources
- reduced hallucination risk

#### Simple Example

Instead of asking the model:

"What does our company refund policy say?"

you first retrieve the actual policy document and then ask the model to answer using that text.

#### Stage 4: Tool Calling

#### The Problem

A model can describe actions, but that is not the same as actually doing them.

#### The Shift

Tool calling lets the model request that the surrounding system run a real function.

Examples:

- search the web
- query a database
- read a file
- create a calendar event

#### Why It Mattered

This turned AI systems from pure text generators into action-capable software.

#### Simple Example

If the user asks:

"What is the weather in San Francisco?"

the model can call a weather tool instead of hallucinating a weather answer.

#### Stage 5: Agents

#### The Problem

Some tasks need multiple steps, not one answer.

#### The Shift

Agentic systems let the model decide what to do next across several steps:

- plan
- retrieve
- call tools
- inspect results
- continue or stop

#### Why It Mattered

This opened up more complex tasks such as research, debugging, and workflow automation.

#### Important Caveat

Agents are often overused.

Many tasks are better as normal workflows with fixed steps.

#### Stage 6: MCP and Standardized Integrations

#### The Problem

As tools grew, many integrations were custom and inconsistent.

#### The Shift

Protocols such as MCP provide a more standardized way for models, tools, and resources to connect.

#### Why It Mattered

This reduces one-off glue code and makes integrations more portable.

#### Stage 7: Evals and Reliability

#### The Problem

A few impressive outputs can hide a fragile system.

#### The Shift

Teams started treating evals, guardrails, and observability as core parts of AI system design.

#### Why It Mattered

This is the difference between:

- a cool demo
- a system you can trust and improve

#### What To Remember From This Section

- instruction tuning made models more usable
- alignment improved assistant behavior
- RAG helped ground answers in real data
- tool calling added real actions
- agents enabled multi-step behavior
- MCP helped standardize integrations
- evals became necessary for serious systems
