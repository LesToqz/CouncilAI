# CouncilAI

CouncilAI is a local multi-LLM debate assistant. It attaches to ChatGPT, Gemini, and Claude tabs in a controllable Microsoft Edge window, sends prompts through their consumer chat UIs, lets the models critique and refine each other, and returns a final synthesized answer.

## Important Warning

CouncilAI depends on browser automation of consumer AI websites. It does not bypass login, CAPTCHA, payment, or access controls. Users must manually log into their own ChatGPT, Gemini, and Claude accounts. This project is intended for local personal use and research/demo purposes.

Website UIs may change and break automation. Use responsibly and respect each service's terms and limits.

## What It Does

- Runs a local Gradio app at `http://127.0.0.1:7860`.
- Uses Microsoft Edge through Playwright.
- Attaches to AI tabs you already opened in the same Edge window.
- Supports Normal mode and Debate mode.
- Saves complete run logs to JSONL and SQLite.
- Continues when one model fails, as long as enough selected models remain usable.

## What It Does Not Do

- It does not store passwords.
- It does not automate username, password, CAPTCHA, payment, or access-control flows.
- It does not bypass login, CAPTCHA, payment, usage limits, or access controls.
- It does not use paid model APIs.
- It does not use Electron, React, Node.js, Docker, or TypeScript.

## Setup

Use Windows PowerShell from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install msedge
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

## Browser Setup

Start the app:

```powershell
python app.py
```

The app automatically opens a separate controllable Edge window with the CouncilAI tab. In that same Edge window, manually open:

- `https://chatgpt.com/`
- `https://gemini.google.com/`
- `https://claude.ai/`

Use those tabs normally and log in manually if needed. CouncilAI does not open AI login tabs and does not automate login.

Remote debugging lets local programs control that Edge instance. Use it only on your own machine and close that Edge window when you are done.

## Running CouncilAI

1. Enter a prompt.
2. Choose Normal or Debate.
3. Select 1 to 5 debate iterations for Debate mode.
4. Enable the model checkboxes you want to use.
5. Optionally click **Check Existing AI Tabs**.
6. Click **Execute**.

Normal mode sends the prompt once to the selected AI tabs and shows their initial outputs side by side. It does not run critique, refinement, or final synthesis.

Debate mode shows the live side-by-side council board with initial answers, critiques, refinements, errors, and final synthesis.

## Logs

JSONL logs are written to:

```text
data/debate_logs/
```

SQLite logs are written to:

```text
data/sqlite/debates.db
```

Browser profiles and logs are ignored by Git because they may contain private session data. The controllable Edge window uses `profiles/app_edge_profile`.

## Known Limitations

- Chat website selectors are fragile and may need updates when sites change.
- Login sessions can expire.
- CAPTCHA or anti-bot checks require manual user action.
- The MVP uses fixed iteration counts; semantic early stopping is stubbed for later.
- Browser automation can time out if a model streams slowly or the website changes layout.

## Troubleshooting

- If CouncilAI cannot attach to Edge, close the Edge window opened by CouncilAI and restart `python app.py`.
- If a model says it is not logged in, log in manually in that existing Edge tab.
- If a tab is not found, open ChatGPT, Gemini, or Claude in the same Edge window that contains the CouncilAI tab.
- If a prompt box cannot be found, the website layout likely changed. Update selectors in `src/browser/*_adapter.py`.
- If fewer than two selected models work, the app stops and returns a clear error.
