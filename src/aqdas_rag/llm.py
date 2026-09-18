"""Model access for the baseline comparison, with three interchangeable routes.

The baseline needs *a* model to turn passages into a cited answer. Which one is
a billing question, not a design question, so it is configuration:

**claude_cli** -- shells out to the installed `claude -p`. This runs under a
Claude subscription rather than API credits, because the Anthropic API and the
claude.ai subscription are separate billing systems and a Pro/Max plan grants no
programmatic API access. Slower (a process per call) and free if you already pay
for the subscription.

**openrouter** -- any model on OpenRouter via its OpenAI-compatible endpoint.

**anthropic** -- the SDK, needing ANTHROPIC_API_KEY from console.anthropic.com,
which is pay-as-you-go credits.

For measuring whether *retrieved context* supports a correctly cited answer,
the identity of the model is not the variable under test -- both arms of the
comparison use the same one, and only the context differs. So picking whichever
route costs nothing is a legitimate choice rather than a compromise.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CLI_TIMEOUT_S = 180


def load_dotenv() -> None:
    """Read .env into the environment, without a dependency.

    Keys belong in a gitignored file rather than in the user environment --
    `setx` writes a credential permanently into the Windows profile, where it
    leaks into every process started thereafter.
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class ModelUnavailable(RuntimeError):
    """Raised when the selected route cannot run, with what to do about it."""


class ClaudeCLI:
    """Drives the installed `claude` binary in print mode."""

    name = "claude-cli"

    def __init__(self, model: str | None = None) -> None:
        self.binary = shutil.which("claude")
        if not self.binary:
            raise ModelUnavailable(
                "`claude` is not on PATH. Install Claude Code, or choose another route."
            )
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        cmd = [self.binary, "-p", "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        # The system prompt is prepended to the user turn rather than passed as
        # a flag: print mode's system-prompt flags have varied across versions,
        # and the contract here is only "the model sees these instructions".
        payload = f"{system}\n\n---\n\n{prompt}"

        result = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CLI_TIMEOUT_S,
        )
        if result.returncode != 0:
            raise ModelUnavailable(
                f"claude -p exited {result.returncode}: {result.stderr.strip()[:300]}"
            )

        try:
            return json.loads(result.stdout).get("result", "")
        except json.JSONDecodeError:
            return result.stdout


class OpenRouter:
    """Any OpenRouter model over its OpenAI-compatible chat endpoint."""

    name = "openrouter"

    def __init__(self, model: str = "qwen/qwen3-next-80b-a3b-instruct") -> None:
        load_dotenv()
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.key:
            raise ModelUnavailable(
                "OPENROUTER_API_KEY is not set. Put it in a .env file at the "
                "project root as OPENROUTER_API_KEY=sk-or-..."
            )
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 900,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


class AnthropicAPI:
    """The SDK route, on pay-as-you-go credits."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        load_dotenv()
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ModelUnavailable(
                "ANTHROPIC_API_KEY is not set. This route needs API credits from "
                "console.anthropic.com -- a Claude subscription does not grant them. "
                "Use --route claude_cli to run on the subscription instead."
            )
        from anthropic import Anthropic

        self.client = Anthropic(api_key=key)
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        reply = self.client.messages.create(
            model=self.model,
            max_tokens=900,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in reply.content if b.type == "text")


ROUTES = {
    "claude_cli": ClaudeCLI,
    "openrouter": OpenRouter,
    "anthropic": AnthropicAPI,
}


def get_model(route: str, model: str | None = None):
    if route not in ROUTES:
        raise ModelUnavailable(f"Unknown route {route!r}. Choose from {list(ROUTES)}.")
    return ROUTES[route](model) if model else ROUTES[route]()


def main() -> int:
    """Report which routes are usable right now."""
    load_dotenv()
    for name in ROUTES:
        try:
            client = get_model(name)
            print(f"  {name:<12} available  ({client.name})")
        except ModelUnavailable as exc:
            print(f"  {name:<12} unavailable -- {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
