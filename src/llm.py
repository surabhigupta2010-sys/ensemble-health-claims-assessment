import os

from dotenv import load_dotenv

from openai import OpenAI

from config import MODEL_NAME


# ----------------------------
# Initialize Gemini Client
# ----------------------------

load_dotenv(override=True)

GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)

google_api_key = os.getenv("GOOGLE_API_KEY")

client = OpenAI(
    base_url=GEMINI_BASE_URL,
    api_key=google_api_key
)


# ----------------------------
# Prompt Creation
# ----------------------------

def create_prompt(row):
    """
    Creates prompt for claim explanation.
    """

    prompt = f"""
You are an experienced medical claims analyst.

A machine learning model has predicted the denial risk for the following insurance claim.

Generate a concise explanation using ONLY the information provided below.

Do not invent additional facts.

Claim Details

Claim ID: {row['claim_id']}

Payer ID: {row['payer_id']}

Payer Type: {row['payer_type']}

Visit Type: {row['visit_type']}

Total Billed: {row['total_billed']}

Expected Payment: {row['expected_payment']}

Number of Procedures: {row['num_procedures']}

Number of Diagnoses: {row['num_diagnoses']}

Prior Authorization Required: {row['prior_auth_required']}

Prior Authorization Present: {row['has_prior_auth']}

In Network: {row['is_in_network']}

Days to Submit: {row['days_to_submit']}

Missing Documentation: {row['missing_documentation_flag']}

Eligibility Verified: {row['eligibility_verified']}

Referral Required: {row['referral_required']}

Referral Present: {row['referral_present']}

Service Month: {row['service_month']}

Payment Ratio: {row['payment_ratio']:.2f}

Payment Gap: {row['payment_gap']:.2f}

Authorization Missing: {row['auth_missing']}

Referral Missing: {row['referral_missing']}

Late Submission: {row['late_submission']}

Predicted Denial Probability: {row['denial_probability']:.2%}

Predicted Denial: {row['predicted_denial']}

Risk Tier: {row['risk_tier']}

Top Risk Factors:
{row['top_risk_factors']}

Instructions

Explain why the model assigned this level of denial risk.

Reference the important claim attributes where appropriate.

Suggest one practical action to reduce denial risk if applicable.

Keep the explanation under 120 words.

Do not mention SHAP.

Do not mention probabilities unless useful.

Do not invent missing information.
"""

    return prompt


# ----------------------------
# Generate Explanation
# ----------------------------

def generate_explanation(row):
    """
    Generates LLM explanation for one claim.
    """

    prompt = create_prompt(row)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()