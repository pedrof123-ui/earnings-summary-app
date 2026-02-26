import os
import smtplib
from email.message import EmailMessage
import gradio as gr
from dotenv import load_dotenv
from datetime import datetime
# from edgar import Company, set_identity
from openai import AsyncOpenAI
from agents import Agent, Runner, trace, OpenAIChatCompletionsModel, function_tool

from earnings_call_functions import scrape_earnings_call_transcript

load_dotenv(override=True)


# set_identity(os.getenv('SEC_ID', 'equity-report-user'))

openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)

OPENROUTER_MODELS = [
    "x-ai/grok-4.1-fast",
    "anthropic/claude-sonnet-4.5",
    "moonshotai/kimi-k2.5",
    "google/gemini-3-flash-preview",
    "stepfun/step-3.5-flash:free",
    "arcee-ai/trinity-large-preview:free",
    "openai/gpt-5.2",
]


def load_supported_tickers() -> set:
    csv_path = os.path.join(os.path.dirname(__file__), "tickers_to_keep.csv")
    with open(csv_path) as f:
        return {line.strip().upper() for line in f if line.strip()}


SUPPORTED_TICKERS = load_supported_tickers()

# SUMMARY_INSTRUCTIONS_TEMPLATE = """
#     You are an equity financial analyst assistant.
#     Create a concise, investment-grade summary of the earnings call transcript provided
#     and the Management Discussion & Analysis from the corresponding 10K or 10Q.

#     For additional context, use tools to retrieve the MD&A section of the latest 10K report or 10Q report.
    
#     Compare the earnings call financial period with the financial period in the latest MD&A report available.
    
#     IMPORTANT: If the the earnings call financial period doesn't match the financial period in the MD&A report,
#     do not use the MD&A report in the summary and say that financial report not available for the 
#     earnings call period. 

#     If earnings call is for Q4 and the latest 10K statement matches earnings call financial period
#     end date, use the latest 10K MD&A report for additional context.

#     Must include:
#         - Today's date, Earnings call date, Financial Period of the earnings call, Report type (10K or 10Q or not released yet), prepared by AI model name
#         - Key financial metrics (revenue, EPS, margins, guidance)
#         - Brief management commentary on business performance
#         - New Services/Technology/Products and Innovation
#         - Key takeaways and forward-looking statements

#     - DO NOT INCLUDE INVESTMENT ADVICE
    
#     Earnings call transcript: {transcript}
#     Today's date: {date}
#     AI model: {model}
# """
SUMMARY_INSTRUCTIONS_TEMPLATE = """
    You are an equity financial analyst assistant.
    Create an investment-grade summary report of the earnings call transcript provided

    Must include:
        - Today's date, Earnings call date, Financial Period of the earnings call, prepared by AI model name
        - Key financial metrics (revenue, EPS, margins, guidance)
        - Brief management commentary on business performance
        - New Services/Technology/Products and Innovation
        - Key takeaways and forward-looking statements

    - DO NOT INCLUDE INVESTMENT ADVICE
    
    Earnings call transcript: {transcript}
    Today's date: {date}
    AI model: {model}
"""

async def generate_report(ticker: str, model_name: str) -> tuple[str, str]:
    ticker = ticker.strip().upper()

    if not ticker:
        return "Please enter a ticker symbol.", ""

    if ticker not in SUPPORTED_TICKERS:
        return f"Ticker '{ticker}' is not supported. Please enter a different ticker symbol.", ""

    transcript = await scrape_earnings_call_transcript(ticker)
    raw_data = {}

    # @function_tool
    # def get_tenk(company: str):
    #     """
    #     Fetches the Management Discussion & Analysis of the latest 10K report from SEC Edgar.

    #     Args:
    #         company: Stock ticker symbol (e.g. 'PFE', 'AAPL', 'NVDA').

    #     Returns:
    #         Returns the fiscal year and MD&A section of the 10K report.
    #     """
    #     print("retrieving 10K report .... ")
    #     tenk = Company(company).get_filings(form="10-K")[0].obj()
    #     fiscal_year = tenk.period_of_report
    #     mda_10k = tenk.management_discussion
    #     raw_data['form'] = '10-K'
    #     raw_data['period'] = str(fiscal_year)
    #     raw_data['mda'] = str(mda_10k)
    #     print("typing the report... ")
    #     return fiscal_year, mda_10k

    # @function_tool
    # def get_tenq(company: str):
    #     """
    #     Fetches the Management Discussion & Analysis of the latest 10Q report from SEC Edgar.

    #     Args:
    #         company: Stock ticker symbol (e.g. 'PFE', 'AAPL', 'NVDA').

    #     Returns:
    #         Returns the Quarter and MD&A section of the 10Q report.
    #     """
    #     print("retrieving 10Q report .... ")
    #     tenq = Company(company).get_filings(form="10-Q")[0].obj()
    #     quarter = tenq.period_of_report
    #     mda_10q = tenq["Item 2"]
    #     raw_data['form'] = '10-Q'
    #     raw_data['period'] = str(quarter)
    #     raw_data['mda'] = str(mda_10q)
    #     print("typing the report... ")
    #     return quarter, mda_10q

    model = OpenAIChatCompletionsModel(model=model_name, openai_client=openrouter_client)
    instructions = SUMMARY_INSTRUCTIONS_TEMPLATE.format(
        transcript=transcript,
        date=datetime.now().strftime("%Y-%m-%d"),
        model=model_name,
    )

    agent = Agent(
        name="earnings_call_reporter",
        instructions=instructions,
        model=model,
        #tools=[get_tenk, get_tenq],
    )

    with trace("Earnings_call_reporter"):
        result = await Runner.run(
            agent,
            f"Create an investment grade report of the latest earnings call of {ticker}",
        )

    sep = "=" * 60
    raw_right = f"EARNINGS CALL TRANSCRIPT\n{sep}\n\n{transcript}"
    if raw_data:
        raw_right += (
            f"\n\n{sep}\n"
            f"{raw_data['form']} MD&A  —  Period: {raw_data['period']}\n"
            f"{sep}\n\n{raw_data['mda']}"
        )

    return result.final_output, raw_right


def send_feedback(feedback_text: str):
    if not feedback_text.strip():
        return "Please enter a feedback message before submitting.", gr.Group(visible=True)
    try:
        msg = EmailMessage()
        msg["From"] = os.getenv("SMTP_FROM")
        msg["To"] = os.getenv("SMTP_TO")
        msg["Subject"] = "Earnings Summary App — User Feedback"
        msg.set_content(feedback_text)

        with smtplib.SMTP_SSL(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT", 465))) as smtp:
            smtp.login(os.getenv("SMTP_USERNAME"), os.getenv("SMTP_PASSWORD"))
            smtp.send_message(msg)

        return "Thank you! Your feedback has been sent.", gr.Group(visible=False)
    except Exception as e:
        return f"Error sending feedback: {e}", gr.Group(visible=True)


PANEL_CSS = """
#summary-panel, #raw-panel {
    height: 700px;
    overflow-y: auto;
}
#summary-panel .prose,
#summary-panel .md-content {
    overflow-y: visible;
}
#feedback-btn {
    position: fixed;
    top: 16px;
    right: 16px;
    z-index: 1000;
    width: auto !important;
    min-width: 90px !important;
}
#feedback-panel {
    position: fixed;
    top: 60px;
    right: 16px;
    z-index: 999;
    width: 380px;
    background: var(--background-fill-primary, white);
    border: 1px solid var(--border-color-primary, #ccc);
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}
"""

with gr.Blocks(title="Earnings Call AI Summary Generator") as demo:
    gr.Markdown("# Earnings Call AI Summary Generator")
    gr.Markdown(
        "<small>Generates an AI-written investment-grade summary of the latest earnings call transcript. "
        "Enter a ticker symbol, select a model, and click **Generate Report**. "
        "The left panel shows the AI summary; the right panel shows the raw transcript.</small><br>"
        "<small>⏱ Depending on the selected LLM, processing can take longer than 1 minute or more.</small>"
    )
    with gr.Row():
        ticker_input = gr.Textbox(label="Company Ticker Symbol", placeholder="e.g. AAPL, NVDA, MSFT")
        model_dropdown = gr.Dropdown(choices=OPENROUTER_MODELS, value=OPENROUTER_MODELS[0], label="Choose LLM Model for AI Summary")
    generate_btn = gr.Button("Generate Report", variant="primary")
    feedback_btn = gr.Button("Feedback / Contact me", variant="secondary", size="sm", elem_id="feedback-btn")

    with gr.Group(visible=False, elem_id="feedback-panel") as feedback_panel:
        gr.Markdown("### Submit Feedback")
        feedback_input = gr.Textbox(
            label="Your Feedback / Contact Me",
            placeholder="Enter your message here...",
            lines=5,
        )
        feedback_status = gr.Markdown("")
        with gr.Row():
            cancel_btn = gr.Button("Cancel", variant="secondary")
            submit_feedback_btn = gr.Button("Submit", variant="primary")

    feedback_btn.click(lambda: gr.Group(visible=True), outputs=feedback_panel)
    cancel_btn.click(lambda: (gr.Group(visible=False), "", ""), outputs=[feedback_panel, feedback_input, feedback_status])
    submit_feedback_btn.click(
        fn=send_feedback,
        inputs=feedback_input,
        outputs=[feedback_status, feedback_panel],
    )

    with gr.Row():
        summary_output = gr.Markdown(
            label="AI Summary",
            show_label=True,
            elem_id="summary-panel",
        )
        raw_output = gr.Textbox(
            label="Raw earnings call transcript",
            lines=30,
            max_lines=30,
            interactive=False,
            elem_id="raw-panel",
        )
    generate_btn.click(fn=generate_report, inputs=[ticker_input, model_dropdown], outputs=[summary_output, raw_output])

if __name__ == "__main__":
    demo.launch(css=PANEL_CSS)
