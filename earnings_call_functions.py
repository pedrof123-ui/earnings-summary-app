import os
import re
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from agents.mcp import MCPServerStdio
from playwright.async_api import async_playwright
from markdownify import markdownify as md_convert

load_dotenv(override=True)

PLAYWRIGHT_PARAMS = {
    "command": "/home/pedro/.nvm/versions/node/v22.22.0/bin/npx",
    "args": [
        "@playwright/mcp@latest",
        "--browser", "chromium",
        "--headless",
        "--timeout-action", "15000",
    ],
}


#@function_tool
async def get_earnings_call_summary(company: str) -> str:
    """
    Fetches and summarizes the latest earnings call transcript for a given company ticker
    from Yahoo Finance. Also saves the summary as a markdown file in the sandbox directory.

    Args:
        company: Stock ticker symbol (e.g. 'PFE', 'AAPL', 'NVDA').

    Returns:
        Markdown-formatted summary of the latest earnings call including key financial
        metrics, management commentary, strategic outlook, and notable Q&A highlights.
    """
    url = f"https://finance.yahoo.com/quote/{company}/earnings-calls/"
    sandbox_path = os.path.abspath(os.path.join(os.getcwd(), "sandbox"))

    files_params = {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", sandbox_path],
    }

    instructions = f"""
You are a financial analyst assistant.

Workflow:
1) Navigate directly to the URL below.
2) Find and click the latest earnings call transcript link.
3) Extract and summarize the most relevant items:
   - Key financial metrics (revenue, EPS, margins, guidance)
   - Management commentary on business performance
   - Strategic priorities and forward-looking statements
   - Notable Q&A highlights
4) Save the markdown summary to {company}_Q#_FY_year_earnings_call.md using the filesystem tool.
5) Return the full markdown summary as your final response.

If a click fails once, do not retry; proceed with snapshot-only extraction.

URL = {url}
"""

    async with MCPServerStdio(params=files_params, client_session_timeout_seconds=60) as mcp_files:
        async with MCPServerStdio(params=PLAYWRIGHT_PARAMS, client_session_timeout_seconds=60) as mcp_browser:
            agent = Agent(
                name="earnings_call_fetcher",
                instructions=instructions,
                model="gpt-5.2",
                mcp_servers=[mcp_files, mcp_browser],
            )

            result = await Runner.run(
                agent,
                f"Find and summarize the latest earnings call of {company}, "
                "then return the full markdown summary.",
                max_turns=30,
            )

            return result.final_output


# ---------------------------------------------------------------------------
# Direct-scraping variant (no agent / no LLM)
# ---------------------------------------------------------------------------

def _parse_quarter_year(text: str) -> tuple[str, str]:
    """Return (quarter, year) parsed from arbitrary text, e.g. ('Q1', '2025')."""
    quarter_words = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}

    q_match = re.search(r"\b(Q[1-4])\b", text, re.IGNORECASE)
    quarter = q_match.group(1).upper() if q_match else None

    if not quarter:
        word_match = re.search(
            r"\b(first|second|third|fourth)\s+quarter\b", text, re.IGNORECASE
        )
        quarter = (
            quarter_words.get(word_match.group(1).lower(), "Q?") if word_match else "Q?"
        )

    year_match = re.search(r"\b(20\d{2})\b", text)
    year = year_match.group(1) if year_match else str(date.today().year)
    return quarter, year


async def scrape_earnings_call_transcript(
    company: str, output_dir: str = "reports"
) -> str:
    """
    Scrapes the latest earnings call transcript for a given company ticker
    from Yahoo Finance and saves it as a markdown file.

    Uses Playwright directly (no agent / no LLM) so the raw transcript is
    preserved rather than summarised.

    Args:
        company:    Stock ticker symbol (e.g. 'AAPL', 'NVDA').
        output_dir: Directory to save the markdown file (default: 'reports').

    Returns:
        Markdown-formatted raw transcript content.
    """
    listing_url = f"https://finance.yahoo.com/quote/{company}/earnings-calls/"
    print("Listening to earnings call.... ")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        page = await context.new_page()

        # 1. Load the earnings-calls listing page
        await page.goto(listing_url, wait_until="domcontentloaded", timeout=30_000)
        await page.wait_for_timeout(4_000)  # allow JS to render

        # 2. Find and navigate to the latest transcript
        transcript_url = listing_url
        for selector in [
            "a[href*='transcript']",
            "a:has-text('Transcript')",
            "button:has-text('Transcript')",
            "a:has-text('Read transcript')",
            "a:has-text('View transcript')",
        ]:
            try:
                locator = page.locator(selector).first
                href = await locator.get_attribute("href", timeout=3_000)
                if href:
                    transcript_url = (
                        f"https://finance.yahoo.com{href}"
                        if href.startswith("/")
                        else href
                    )
                    await page.goto(
                        transcript_url, wait_until="domcontentloaded", timeout=30_000
                    )
                    await page.wait_for_timeout(4_000)
                    break
                # href-less element — click in place
                await locator.click(timeout=3_000)
                await page.wait_for_timeout(4_000)
                break
            except Exception:
                continue

        page_title = await page.title()
        final_url = page.url

        # 3. Extract transcript body HTML (widest-to-narrowest fallback)
        content_html = ""
        for selector in [
            "article",
            "main",
            "[data-testid*='transcript']",
            ".transcript",
            "body",
        ]:
            try:
                element = page.locator(selector).first
                if await element.is_visible(timeout=3_000):
                    content_html = await element.inner_html()
                    break
            except Exception:
                continue

        await browser.close()

    # 4. HTML → clean markdown
    raw_md = md_convert(
        content_html,
        heading_style="ATX",
        strip=["script", "style", "nav", "footer", "header"],
    )
    copyright_marker = "Copyright ©"
    cut = raw_md.find(copyright_marker)
    if cut != -1:
        raw_md = raw_md[:cut]
    clean_md = re.sub(r"\n{3,}", "\n\n", raw_md).strip()

    # 5. Parse quarter + year for the filename
    quarter, year = _parse_quarter_year(page_title + "\n" + clean_md[:1_500])

    # 6. Build the final document with a metadata header
    header = (
        f"# {company} {quarter} FY{year} Earnings Call Transcript\n\n"
        f"**Source:** {final_url}\n\n"
        f"**Scraped:** {date.today().isoformat()}\n\n"
        "---\n\n"
    )
    full_content = header + clean_md

    # 7. Save to file
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"{company}_{quarter}_FY{year}_earnings_call.md"
    out_file.write_text(full_content, encoding="utf-8")

    print("Got it!")

    return full_content
