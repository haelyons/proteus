Our working directory is /home/hal/dev/interp. Read autolabel/BRIEF.md first; it is the context I'd otherwise give you in person, and it says what to read and what to leave alone.

Your /goal is to understand what it would mean to apply Cammarata's approach ("Neural Networks Are Already Clean", Goodfire, 2026-10-15; full text in janus/authors/informed.prompt_sent.md) to automated unit labelling, and where parallelism (harnesses, kernels, batching, scheduling) changes what is possible rather than only making it faster.

The questions:
1. What is his approach, stated precisely? What does it claim, what does it measure, and what result would show it failing? Separate what a unit responds to from how it is wired, and say which of his steps address each.
2. What do current automated labelling pipelines do (nnsight and the tooling around it, Goodfire's 1000 published neuron investigations, the field's auto-interp methods), and where does each spend its compute? Which steps are embarrassingly parallel, which are sequential, and why?
3. What did janus establish, round by round, about whether a model can read a unit the way he does? Which findings are about the reader, which about the data it was shown, and which about the unit?
4. What would recovering his labels look like as a calibration, on a model he studied, and how would we know a harness reads units his way rather than assuming they are clean?
5. Where would a better harness or kernel matter most: collecting activations, the reading loop, or the checks (held-out prediction, near misses, families, downstream readers)?

Before answering, register your prediction for each question in autolabel/PREDICTIONS.md: what you expect the answer to be, and why. Then look at the available evidence and make observations. Don't update your understanding until you've observed as thoroughly as possible without updating it. Read the essay and janus's README before anything else, then the tools behind the rounds you cite, not all at once.

Refer to the evidence clearly: files with line numbers, commits, URLs. State uncertainties clearly, cite exhaustively, and ask why. Explain yourself as simply as possible, but no simpler.

Leave biology out of this pass, as the brief describes; if a question needs it, stop and tell me. No GPU launches, spend, commits or pushes without asking. Write only under autolabel/.

Do not suggest solutions until you've discussed and thought about the problem as thoroughly as possible. Don't answer until you've thought about the problem as thoroughly as possible without answering.
