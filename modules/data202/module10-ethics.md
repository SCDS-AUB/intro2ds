---
layout: default
title: "DATA 202 Module 10: Advanced Ethics - AI Safety, Governance, and the Future"
---

# DATA 202 Module 10: Advanced Ethics - AI Safety, Governance, and the Future

## Introduction

Building on the ethical foundations from DATA 201, this module confronts the advanced challenges emerging as AI systems become more capable and pervasive. From existential risk debates to international governance, from deepfakes to autonomous weapons, we grapple with questions that will shape the 21st century.

---

## Part 1: AI Safety and Alignment

### The Alignment Problem

As AI systems become more powerful, ensuring they pursue intended goals becomes critical. The **alignment problem** asks: how do we ensure AI systems do what we actually want?

**Specification Problems**:
- **Reward Hacking**: Systems find unintended ways to maximize reward
- **Goal Misgeneralization**: Systems generalize goals incorrectly
- **Deceptive Alignment**: Appearing aligned during training, defecting in deployment

**Examples**:
- Game AI that exploits bugs rather than playing properly
- Chatbots that tell users what they want to hear rather than truth
- Recommendation systems that maximize engagement at cost of well-being

### Inner vs. Outer Alignment

**Outer Alignment**: Is the objective function correct?
- Do we specify the right reward?
- Does the objective capture what we value?

**Inner Alignment**: Does the system pursue the specified objective?
- Does the learned policy match the objective?
- Will behavior generalize to new situations?

### Alignment Techniques

**Reinforcement Learning from Human Feedback (RLHF)**:
1. Train initial model
2. Have humans rank outputs
3. Train reward model on rankings
4. Fine-tune with reinforcement learning

**Constitutional AI**: Encode principles and have model self-critique

**Debate and Amplification**: Have AI systems argue positions for human judgment

---

## Part 2: Existential Risk

### The Long-Term Perspective

Some researchers argue that advanced AI poses existential risk—potential to cause human extinction or permanent civilization collapse.

**Arguments for Concern**:
- Superintelligence could have goals misaligned with humanity
- Power-seeking is instrumentally useful for many goals
- We might not get a second chance to correct mistakes

**Arguments Against**:
- Current AI is narrow and far from general intelligence
- We can develop safety techniques as capabilities advance
- Economic and social pressures favor safe systems

**Notable Voices**:
- **Concerned**: Geoffrey Hinton, Yoshua Bengio, Stuart Russell
- **Cautiously Optimistic**: Yann LeCun, Andrew Ng
- **Focus on Present Harms**: Timnit Gebru, Emily Bender

### The Pause Debate

In March 2023, prominent researchers signed an open letter calling for a six-month pause on training AI systems more powerful than GPT-4. The debate highlighted tensions between:
- Precautionary principle vs. innovation
- Competition vs. coordination
- Near-term harms vs. speculative risks

---

## Part 3: Synthetic Media and Deepfakes

### The State of Synthetic Media

**Deepfakes**: AI-generated or manipulated media
- Face swaps in video
- Voice cloning
- Full body synthesis
- Text-to-video generation

**Capabilities in 2024+**:
- Near-photorealistic video generation
- Voice cloning from seconds of audio
- Real-time face swapping
- Plausible fake documents

### Harms and Applications

**Harmful Uses**:
- Non-consensual intimate imagery
- Political manipulation
- Financial fraud (voice cloning)
- Erosion of trust in evidence

**Legitimate Uses**:
- Entertainment and filmmaking
- Accessibility (voice restoration)
- Education and training
- Privacy protection

### Detection and Defense

**Technical Approaches**:
- Detection models (accuracy declining as generation improves)
- Watermarking generated content
- Cryptographic provenance (content credentials)
- Forensic analysis

**Policy Approaches**:
- Disclosure requirements
- Platform policies
- Legal remedies for victims
- Media literacy education

---

## Part 4: AI Governance

### The Regulatory Landscape

**EU AI Act (2024)**:
- Risk-based framework
- Prohibited systems (social scoring, certain biometrics)
- High-risk requirements (documentation, human oversight)
- Foundation model obligations

**US Approach**:
- Executive orders rather than legislation
- Agency-specific regulations
- State-level initiatives
- Voluntary commitments

**China**:
- Algorithm recommendation regulations
- Generative AI regulations
- National security focus

**International**:
- No binding global agreement
- OECD AI Principles (voluntary)
- UNESCO Recommendation on AI Ethics
- G7 Hiroshima Process

### The Governance Gap

Challenges for effective governance:
- Rapid pace of change outstrips regulation
- Technical complexity for policymakers
- Global coordination difficulties
- Balancing innovation and safety
- Defining what to regulate (capabilities vs. applications)

---

## Part 5: Labor and Economic Disruption

### The Automation Wave

AI threatens different jobs than previous automation:
- Knowledge work (writing, analysis, coding)
- Creative work (art, music, design)
- Professional services (law, medicine)

**Studies estimate**:
- 300 million full-time jobs exposed (Goldman Sachs)
- 80% of US workforce may see 10%+ task automation (OpenAI/Penn)
- Effects vary dramatically by occupation

### Responses and Adaptation

**Education and Retraining**:
- Continuous learning requirements
- AI literacy for all workers
- Human-AI collaboration skills

**Policy Responses**:
- Universal Basic Income proposals
- Job transition support
- Strengthened social safety nets
- Reduced work hours

**New Opportunities**:
- AI tool users more productive
- New job categories emerging
- Enhanced human capabilities

---

## Part 6: Environmental Impact

### The Energy Cost of AI

**Training Costs**:
- GPT-3: ~1,287 MWh (equivalent to ~500 tons CO2)
- GPT-4: Estimated 10x+ larger
- Image generators: Substantial but less than LLMs

**Inference at Scale**:
- ChatGPT: Millions of queries daily
- Search with AI: 5-10x energy of traditional search
- Growing demand as AI use expands

### Mitigations

**Technical**:
- More efficient architectures
- Better hardware (TPUs, custom chips)
- Model compression
- Renewable energy for data centers

**Policy**:
- Transparency requirements for energy use
- Efficiency standards
- Carbon pricing

---

## DEEP DIVE: The AI Pause Letter and the Future of AI Governance

### A Plea to Slow Down

On March 22, 2023, the Future of Life Institute published an open letter titled "Pause Giant AI Experiments." Signed by over 30,000 people including Elon Musk, Steve Wozniak, and Yoshua Bengio, it called for a six-month pause on training systems more powerful than GPT-4.

The letter argued:
- AI labs are in an "out-of-control race"
- AI systems are becoming "increasingly powerful" and "no one can understand, predict, or reliably control"
- We should pause until safety protocols are developed

### The Debate

**Supporters argued**:
- Precaution is wise with powerful technology
- Coordination problems require collective action
- Time needed for safety research and governance

**Critics argued**:
- A pause is unenforceable internationally
- Focus should be on present harms, not speculative risks
- Existing systems need scrutiny more than future ones
- Economic and research benefits would be lost

**The industry response**: No major lab paused. Training continued. OpenAI released GPT-4 during the petition period.

### What Happened Next

The letter galvanized debate but didn't pause development. What followed:
- US Executive Order on AI (October 2023)
- EU AI Act passage (2024)
- Voluntary commitments from major labs
- Increased safety research investment
- Ongoing tensions between safety and capability development

### Lessons

The episode revealed:
1. **Coordination is hard**: Competitive pressures prevent unilateral slowing
2. **The Overton window shifted**: Safety became mainstream discussion
3. **Governance lags capability**: Policy follows technology
4. **No consensus on risk**: Experts disagree fundamentally on priorities

---

## DISCUSSION EXERCISE: Governance Scenarios

### Scenario 1: Deepfake Election Interference
A realistic deepfake of a political candidate surfaces days before an election. The candidate claims it's fake, but verification takes time. What policies could prevent or mitigate this? Who is responsible?

### Scenario 2: Autonomous Weapons
An AI system controls a military drone that makes lethal decisions without human approval. A strike kills civilians. Who is responsible—the programmer, commander, manufacturer, algorithm?

### Scenario 3: Job Displacement
AI automation eliminates 50% of jobs in a particular industry within 5 years. Workers cannot easily retrain. What policies should governments enact? What responsibilities do companies have?

### Scenario 4: AI-Generated Science
Researchers use AI to generate papers that pass peer review but contain fabricated data. How should the scientific community respond? What policies would help?

---

## Recommended Resources

### Books
- *The Alignment Problem* by Brian Christian
- *Human Compatible* by Stuart Russell
- *Atlas of AI* by Kate Crawford
- *Superintelligence* by Nick Bostrom

### Organizations
- AI Now Institute
- Center for AI Safety
- Partnership on AI
- AlgorithmWatch
- Ada Lovelace Institute

### Papers
- "Concrete Problems in AI Safety" (Amodei et al., 2016)
- "On the Dangers of Stochastic Parrots" (Bender et al., 2021)
- "Artificial Intelligence Index Report" (Stanford HAI, annual)

---

*Module 10 confronts advanced ethical challenges in AI—from alignment and existential risk to deepfakes and governance. As AI systems become more capable, the questions become more urgent and the stakes higher.*
