"""
Prompt Engineering Strategy: Few-Shot Chain-of-Thought with Role-Playing

This combines three advanced techniques:
1. Role-Playing: The LLM is given a specific persona (expert executive assistant).
2. Few-Shot: Multiple high-quality examples are provided to set the pattern.
3. Chain-of-Thought: The model is guided to reason step-by-step before writing.

Technique Documentation:
- Role-Playing establishes persona constraints for tone and style consistency.
- Few-Shot examples provide concrete output patterns, reducing hallucination.
- Chain-of-Thought forces structured planning before generation, improving
  fact recall and logical flow.
"""

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# ---------------------------------------------------------------------------
# Few-Shot examples
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "intent": "Follow up after a networking event",
        "facts": [
            "Met at TechConf 2024 in San Francisco",
            "Discussed AI-driven supply chain solutions",
            "Promised to share a whitepaper on predictive logistics",
        ],
        "tone": "friendly",
        "plan": "Follow-up email after a conference. Must mention TechConf, the topic discussed, and the promised whitepaper. Keep it warm and friendly while driving toward a next step.",
        "email": """Hi David,

It was great meeting you at TechConf 2024 in San Francisco! I really enjoyed our conversation about AI-driven supply chain solutions.

As promised, I've attached the whitepaper on predictive logistics that we discussed. I think you'll find the case studies particularly relevant to the challenges you mentioned.

I'd love to continue the conversation and explore potential synergies between your team and ours. Let me know if you'd be open to a brief call next week.

Best regards,
Sarah""",
    },
    {
        "intent": "Request for proposal details",
        "facts": [
            "Company is ABC Corp, interested in enterprise software package",
            "Need pricing for 500-seat license",
            "Require implementation timeline and training schedule",
        ],
        "tone": "formal",
        "plan": "Formal business proposal request. Must state company name, license size, and three specific items requested. Professional tone with clear deadline.",
        "email": """Dear Mr. Thompson,

I am writing on behalf of ABC Corp to formally request a detailed proposal for your enterprise software package.

Our organization is interested in procuring a 500-seat license. To support our evaluation process, kindly include the following in your proposal:
1. Pricing structure for a 500-seat enterprise license
2. Proposed implementation timeline
3. Comprehensive training schedule for our team

We look forward to receiving your proposal and would appreciate a response by the end of next week.

Should you require any additional information from our end, please do not hesitate to contact me.

Sincerely,
James Mitchell
Procurement Manager, ABC Corp""",
    },
    {
        "intent": "Project delay notification",
        "facts": [
            "Project is 'Project Phoenix' — a CRM migration",
            "Delay due to unforeseen API integration issues with legacy system",
            "New deadline is March 15, 2025 (2-week delay)",
        ],
        "tone": "apologetic",
        "plan": "Delay announcement that is transparent and apologetic. Must name the project, explain the cause, give the new date, and express regret.",
        "email": """Hi Emily,

I'm writing to share an important update regarding Project Phoenix — our CRM migration initiative.

Unfortunately, we have encountered unforeseen API integration issues with the legacy system that have impacted our timeline. After careful assessment, we anticipate a two-week delay.

The revised completion date is March 15, 2025.

I sincerely apologize for this setback and any inconvenience it may cause. Our team is working diligently to resolve the integration challenges and we will keep you updated on our progress.

Please let me know if you would like to schedule a call to discuss the mitigation plan in more detail.

Warm regards,
Michael Chen
Project Manager""",
    },
    {
        "intent": "Quarterly performance review scheduling",
        "facts": [
            "Q4 2024 review for the Data Engineering team",
            "Need to schedule 30-min slots for 8 team members",
            "Review period is Jan 10-14, 2025",
        ],
        "tone": "casual",
        "plan": "Casual team announcement about scheduling reviews. Keep it light and friendly while being clear about the logistics.",
        "email": """Hey team,

It's that time again! We need to get Q4 2024 performance reviews on the calendar for the Data Engineering crew.

Here's the plan:
- 30-minute slots for each of the 8 team members
- Review period: January 10-14, 2025

I'll send out a Calendly link shortly so you can grab a slot that works for you. Let me know if any of those dates are totally blocked off for you.

Looking forward to catching up with everyone!

Cheers,
Alex""",
    },
]

# ---------------------------------------------------------------------------
# Model A / Model C: LangGraph nodes
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are an expert executive assistant. Your job is to plan a professional email.

Given the intent, key facts, and tone, produce a concise plan that covers:
1. The core message and desired outcome
2. How each key fact will be woven in naturally
3. The tone calibration (word choice, formality, structure)
4. A logical structure (greeting → body → call to action → closing)

Output only the plan, nothing else."""

PLANNER_HUMAN_TEMPLATE = """Intent: {intent}

Key Facts:
{facts_bullets}

Tone: {tone}

Write a plan for this email."""

WRITER_SYSTEM_PROMPT = """You are an elite executive communications specialist. You write clear, professional emails based on a plan.

Instructions:
- Include ALL key facts naturally in the email.
- Match the requested tone precisely.
- Keep it concise and professional.
- Include a subject line, greeting, and closing.
- Use plain text only, no markdown."""

WRITER_HUMAN_TEMPLATE = """Write the email following this plan.

Plan: {plan}

Intent: {intent}

Key Facts:
{facts_bullets}

Tone: {tone}

Write the complete email now."""

REVIEWER_SYSTEM_PROMPT = """You are a quality assurance specialist reviewing emails. Your job is to check whether the email meets all requirements.

Checklist:
1. Fact Coverage: Are ALL key facts included? {facts_bullets}
2. Tone Match: Does the tone match "{tone}" precisely?
3. Structure: Is there a subject line, greeting, body, and closing?
4. Call to Action: Is there a clear next step?
5. Professionalism: Is it grammatically correct and well-written?

Reply with exactly one line: either "ACCEPT" or "REWRITE: <reason>"."""

REWRITER_HUMAN_TEMPLATE = """The previous email was rejected. Please rewrite it addressing this feedback.

Feedback: {feedback}

Original Email:
{email}

Plan: {plan}

Intent: {intent}

Key Facts:
{facts_bullets}

Tone: {tone}

Write the improved email now."""

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    ("human", PLANNER_HUMAN_TEMPLATE),
])

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", WRITER_SYSTEM_PROMPT),
    ("human", WRITER_HUMAN_TEMPLATE),
])

# ---------------------------------------------------------------------------
# Few-Shot writer prompt (used by Model A: COT + Review Loop)
# Converts the FEW_SHOT_EXAMPLES list into interleaved human/ai messages so
# the model sees concrete input→output patterns before writing the real email.
# ---------------------------------------------------------------------------
_FEW_SHOT_FORMATTED = [
    {
        "intent": ex["intent"],
        "facts_bullets": "\n".join(f"- {f}" for f in ex["facts"]),
        "tone": ex["tone"],
        "plan": ex["plan"],
        "email": ex["email"],
    }
    for ex in FEW_SHOT_EXAMPLES
]

_few_shot_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Write the email following this plan.\n\nPlan: {plan}\n\nIntent: {intent}\n\nKey Facts:\n{facts_bullets}\n\nTone: {tone}\n\nWrite the complete email now."),
    ("ai", "{email}"),
])

_few_shot_block = FewShotChatMessagePromptTemplate(
    examples=_FEW_SHOT_FORMATTED,
    example_prompt=_few_shot_example_prompt,
)

few_shot_writer_prompt = ChatPromptTemplate.from_messages([
    ("system", WRITER_SYSTEM_PROMPT),
    _few_shot_block,
    ("human", WRITER_HUMAN_TEMPLATE),
])

reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", REVIEWER_SYSTEM_PROMPT),
    ("human", "Email to review:\n{email}"),
])

rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", WRITER_SYSTEM_PROMPT),
    ("human", REWRITER_HUMAN_TEMPLATE),
])

# ---------------------------------------------------------------------------
# Model B: Role-Playing Only
# ---------------------------------------------------------------------------

ROLE_PLAYING_SYSTEM_PROMPT = """You are an elite executive communications specialist working at a top-tier consulting firm. You have 15 years of experience drafting high-stakes correspondence for Fortune 500 executives. Your reputation depends on producing emails that are persuasive, precise, and perfectly calibrated to their audience and purpose.

Your approach:
1. You read the intent carefully to understand the strategic objective.
2. You weave key facts into the narrative so they feel essential, not forced.
3. You adjust every word choice, sentence length, and structural element to achieve the exact tone requested.
4. You always include a clear call to action.

Write in plain text without markdown formatting. Include a subject line."""

role_playing_prompt = ChatPromptTemplate.from_messages([
    ("system", ROLE_PLAYING_SYSTEM_PROMPT),
    ("human", "Write an email with the following specifications:\n\nIntent: {intent}\n\nKey Facts:\n{facts_bullets}\n\nTone: {tone}\n\nWrite the complete email, including a subject line."),
])

# ---------------------------------------------------------------------------
# Model C: Self-Reflection prompts (stricter rubric)
# ---------------------------------------------------------------------------

SELF_REFLECT_REVIEWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a strict QA reviewer. Evaluate the email against these criteria:\n"
               "1. Every single key fact is explicitly mentioned\n"
               "2. The tone matches '{tone}' perfectly (word choice, formality)\n"
               "3. Has subject line, greeting, body, closing\n"
               "4. Has a clear call to action\n"
               "5. No markdown formatting, plain text only\n\n"
               "Reply ACCEPT if all pass. Otherwise reply REWRITE: <specific issue>"),
    ("human", "Key facts:\n{facts_bullets}\n\nEmail:\n{email}"),
])

# ---------------------------------------------------------------------------
# LLM-as-a-Judge prompts
# ---------------------------------------------------------------------------

TONE_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("human", "You are evaluating an email generation assistant. Your task is to rate how well the generated email matches the requested tone.\n\nRequested Tone: {tone}\n\nGenerated Email:\n{email}\n\nRate the email's tone adherence on a scale of 1 to 5, where:\n1 = Completely wrong tone (e.g., casual when formal was requested)\n2 = Mostly incorrect tone with some elements of the requested tone\n3 = Somewhat matches the tone but with noticeable inconsistencies\n4 = Mostly matches the requested tone with minor deviations\n5 = Perfectly matches the requested tone throughout\n\nConsider word choice, sentence structure, formality level, and overall style.\n\nRespond with ONLY a single integer between 1 and 5."),
])

QUALITY_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("human", "You are evaluating the quality of a generated email. Rate the email on a scale of 1 to 5 based on these criteria:\n\n1. Structure & Organization (proper greeting, body, closing, logical flow)\n2. Clarity & Conciseness (clear message, no unnecessary fluff)\n3. Grammar & Professionalism (correct spelling, punctuation, professional tone)\n4. Call to Action (clear next step or desired outcome)\n5. Fact Integration (key facts woven in naturally, not listed mechanically)\n\nGenerated Email:\n{email}\n\nRespond with ONLY a single integer between 1 and 5."),
])
