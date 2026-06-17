"""
Email Generator: Orchestrates the generation process for a given model.
"""

from models.base import EmailInput, EmailResult


def generate_email(model, email_input: EmailInput) -> EmailResult:
    raw_output, parsed_email = model.generate(email_input)
    return EmailResult(
        input=email_input,
        raw_output=raw_output,
        parsed_email=parsed_email,
        model_name=model.name(),
        strategy_name=model.strategy_name(),
    )
