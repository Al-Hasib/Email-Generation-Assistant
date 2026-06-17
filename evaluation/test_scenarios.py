"""
10 Test Scenarios with Human Reference Emails.

Each scenario covers a different intent/tone combination to
comprehensively evaluate the email generation assistant.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TestScenario:
    id: int
    intent: str
    facts: List[str]
    tone: str
    reference_email: str


SCENARIOS = [
    TestScenario(
        id=1,
        intent="Follow up after a job interview",
        facts=[
            "Interviewed for Senior Data Scientist position on June 10, 2025",
            "Spoke with hiring manager Priya Sharma and team lead Marcus Chen",
            "Discussed experience with transformer models and NLP pipelines",
            "Mentioned interest in the company's healthcare AI initiatives",
        ],
        tone="empathetic",
        reference_email="""Subject: Follow-Up — Senior Data Scientist Interview

Dear Priya and Marcus,

Thank you again for speaking with me on June 10, 2025 regarding the Senior Data Scientist position. I truly enjoyed our conversation and appreciated the thoughtful questions from both you, Priya, as hiring manager, and you, Marcus, as team lead, about my work with transformer models and NLP pipelines.

Our discussion further reinforced my excitement about the role, particularly regarding your healthcare AI initiatives, which align closely with my interest in applying NLP to meaningful real-world problems as I mentioned during our conversation.

I would be delighted to answer any additional questions you may have as you move forward in the decision process. Please feel free to reach out at any time.

Warmest regards,
Alex Rivera""",
    ),
    TestScenario(
        id=2,
        intent="Request for proposal details",
        facts=[
            "Company is ABC Corp, interested in enterprise software package",
            "Need pricing for 500-seat license",
            "Require implementation timeline and training schedule",
            "Deadline for proposal is end of next week",
        ],
        tone="formal",
        reference_email="""Subject: Request for Proposal — Enterprise Software Package

Dear Mr. Thompson,

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
    ),
    TestScenario(
        id=3,
        intent="Project delay notification",
        facts=[
            "Project is 'Project Phoenix' — a CRM migration",
            "Delay due to unforeseen API integration issues with legacy system",
            "New deadline is March 15, 2025 (2-week delay)",
            "Team is working on mitigation plan",
        ],
        tone="apologetic",
        reference_email="""Subject: Update — Project Phoenix Timeline

Hi Emily,

I'm writing to share an important update regarding Project Phoenix — our CRM migration initiative.

Unfortunately, we have encountered unforeseen API integration issues with the legacy system that have impacted our timeline. After careful assessment, we anticipate a two-week delay.

The revised completion date is March 15, 2025.

I sincerely apologize for this setback and any inconvenience it may cause. Our team is actively developing a mitigation plan and is working diligently to resolve the integration challenges. We will keep you updated on our progress.

Please let me know if you would like to schedule a call to discuss this in more detail.

Warm regards,
Michael Chen
Project Manager""",
    ),
    TestScenario(
        id=4,
        intent="Quarterly performance review scheduling",
        facts=[
            "Q4 2024 review for the Data Engineering team",
            "Need to schedule 30-min slots for 8 team members",
            "Review period is Jan 10-14, 2025",
            "Will send Calendly link for slot selection",
        ],
        tone="casual",
        reference_email="""Subject: Q4 2024 Performance Reviews — Let's Schedule!

Hey team,

It's that time again! We need to get Q4 2024 performance reviews on the calendar for the Data Engineering crew.

Here's the plan:
- 30-minute slots for each of the 8 team members
- Review period: January 10-14, 2025

I'll send out a Calendly link shortly so you can grab a slot that works for you. Let me know if any of those dates are totally blocked off.

Looking forward to catching up with everyone!

Cheers,
Alex""",
    ),
    TestScenario(
        id=5,
        intent="Declining a business partnership offer",
        facts=[
            "Offered by InnovateTech for a joint marketing partnership",
            "Decision due to strategic shift toward in-house campaigns",
            "Grateful for the offer and interested in future opportunities",
            "Willing to stay in touch for potential collaboration later",
        ],
        tone="courteous",
        reference_email="""Subject: Regarding Your Partnership Proposal

Dear Sarah,

Thank you so much for offering InnovateTech's joint marketing partnership proposal. We truly appreciate the time and thought that went into it and are grateful for your interest in collaborating with us.

After careful consideration, we have decided to decline the offer at this time. This decision comes as a result of a strategic shift toward building our marketing campaigns in-house.

Please know that this was not an easy decision, and we hold InnovateTech in high regard. I would very much like to stay in touch and remain open to exploring potential collaboration opportunities in the future.

Wishing you and your team continued success.

Warm regards,
David Kim
Marketing Director""",
    ),
    TestScenario(
        id=6,
        intent="Announcing a new product launch",
        facts=[
            "Product is 'DataVue Analytics Pro' — an AI-powered analytics platform",
            "Launch date is September 15, 2025",
            "Early bird pricing available until August 31, 2025",
            "Exclusive demo webinar on August 20, 2025",
        ],
        tone="enthusiastic",
        reference_email="""Subject: Introducing DataVue Analytics Pro — The Future of Analytics is Here!

Hi everyone,

We are thrilled to announce the launch of DataVue Analytics Pro — our next-generation AI-powered analytics platform!

After months of hard work and innovation, we're proud to bring you a tool that will transform how you derive insights from your data.

Here's what you need to know:
- Official Launch Date: September 15, 2025
- Early Bird Pricing: Available until August 31, 2025 (save 30%!)
- Exclusive Demo Webinar: Join us on August 20, 2025 for a first look

Don't miss this opportunity to be among the first to experience DataVue Analytics Pro. Reserve your spot for the webinar today!

Let's unlock the power of your data together.

Cheers,
The DataVue Team""",
    ),
    TestScenario(
        id=7,
        intent="Request time off / vacation approval",
        facts=[
            "Requesting 5 working days from July 21-25, 2025",
            "Reason is family vacation",
            "Will ensure all project deliverables are completed beforehand",
            "Happy to have a handover call before leaving",
        ],
        tone="professional",
        reference_email="""Subject: Time Off Request — July 21-25, 2025

Dear Manager,

I am writing to formally request time off for 5 working days, from July 21 to July 25, 2025, for a family vacation.

I have reviewed my current project commitments and will ensure that all deliverables are completed prior to my leave. I am also happy to schedule a handover call to brief any colleagues who may need to cover responsibilities during my absence.

Please let me know if you require any additional information or if there are any concerns regarding the timing.

Thank you for your consideration.

Best regards,
Jordan""",
    ),
    TestScenario(
        id=8,
        intent="Customer complaint response",
        facts=[
            "Customer reported delayed shipment (Order #45219)",
            "Order was delayed by 5 days due to carrier issues",
            "Offering 20% discount on next order as compensation",
            "Shipment has arrived; customer satisfaction is priority",
        ],
        tone="empathetic",
        reference_email="""Subject: Our Sincerest Apologies — Order #45219

Dear Mrs. Williams,

Thank you for reaching out regarding your recent experience with Order #45219. I am truly sorry to hear about the delay and understand how frustrating this must have been.

After looking into the matter, I can confirm that your shipment was delayed by 5 days due to an issue with our carrier partner. Please know that this does not reflect the level of service we strive to provide, and I sincerely apologize for the inconvenience.

I am pleased to confirm that your order has now arrived. As a gesture of goodwill, we have applied a 20% discount to your next order. You will receive the discount code via email shortly.

Your satisfaction is our top priority, and we are taking steps to prevent similar issues in the future.

Please don't hesitate to contact us if there is anything else we can do.

With sincere apologies,
Emma Roberts
Customer Support Lead""",
    ),
    TestScenario(
        id=9,
        intent="Internal team meeting summary and action items",
        facts=[
            "Sprint review meeting held on June 13, 2025",
            "Three action items: update API docs, fix login bug, prepare Q3 roadmap",
            "Owner for API docs is Sarah, deadline June 20",
            "Owner for login bug is Raj, deadline June 18",
            "Owner for Q3 roadmap is Tom, deadline June 25",
        ],
        tone="professional",
        reference_email="""Subject: Sprint Review Summary — June 13, 2025

Hi team,

Thank you for a productive sprint review meeting on June 13, 2025. Here is a summary of the key action items and owners:

1. Update API Documentation
   Owner: Sarah
   Deadline: June 20, 2025

2. Fix Login Bug
   Owner: Raj
   Deadline: June 18, 2025

3. Prepare Q3 Roadmap
   Owner: Tom
   Deadline: June 25, 2025

Please ensure that any blockers are raised in our daily standup. Let me know if you need any support to meet these deadlines.

Thanks,
Riley""",
    ),
    TestScenario(
        id=10,
        intent="Networking outreach to industry peer",
        facts=[
            "Admire the recipient's recent talk on 'Scaling ML Pipelines' at DataSummit 2025",
            "Work on similar challenges at current company (TechFlow Inc.)",
            "Would like to schedule a 20-min virtual coffee chat",
            "Available on Thursdays or Fridays",
        ],
        tone="friendly",
        reference_email="""Subject: Loved Your Talk at DataSummit 2025!

Hi Dr. Patel,

I recently attended your talk on 'Scaling ML Pipelines' at DataSummit 2025, and I wanted to reach out to say how much I enjoyed it. Your insights on distributed training optimization were particularly valuable.

I work on similar challenges here at TechFlow Inc., and your perspective resonated deeply with our own experiences. I would love to connect and learn more about your work.

Would you be open to a 20-minute virtual coffee chat? I'm generally available on Thursdays or Fridays, but I'm happy to work around your schedule.

Looking forward to hearing from you!

Best,
Kevin Lee
ML Engineer, TechFlow Inc.""",
    ),
]
