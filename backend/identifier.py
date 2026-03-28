"""Card identification agent.

Uses Claude's vision API to identify sports cards from images.
Falls back to manual identification if no API key is available.
"""

import base64
import json
import os
import subprocess
from pathlib import Path


def identify_card_with_claude(image_path: str) -> dict:
    """Use Claude CLI to identify a card from its image.

    Sends the card image to Claude with a structured prompt
    to extract card details and estimate value.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return {"error": f"Image not found: {image_path}"}

    prompt = """You are a vintage sports card expert appraiser. Analyze this card image and provide:

1. **Player Name** — full name of the player
2. **Year** — year the card was produced
3. **Card Set** — the set/series name (e.g., "Topps", "Bowman", "Fleer", "Upper Deck")
4. **Card Number** — the card number if visible
5. **Manufacturer** — card manufacturer
6. **Sport** — baseball, basketball, football, hockey, etc.
7. **Condition** — estimate: Mint, Near Mint, Excellent, Very Good, Good, Fair, Poor
8. **Condition Notes** — any visible damage, wear, centering issues
9. **Rarity** — Common, Uncommon, Rare, Very Rare, Ultra Rare
10. **Estimated Price Low** — conservative estimate in USD
11. **Estimated Price High** — optimistic estimate in USD
12. **Confidence** — your confidence level 0.0 to 1.0

Respond ONLY with valid JSON (no markdown, no explanation):
{
    "player_name": "...",
    "year": 1952,
    "card_set": "...",
    "card_number": "...",
    "manufacturer": "...",
    "sport": "...",
    "condition": "...",
    "condition_notes": "...",
    "rarity": "...",
    "estimated_price_low": 0.00,
    "estimated_price_high": 0.00,
    "ai_confidence": 0.0
}"""

    try:
        # Use Claude CLI with image input
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json", str(image_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return {"error": f"Claude CLI failed: {result.stderr}"}

        # Parse the response — Claude returns JSON with a "result" field
        response = json.loads(result.stdout)
        text = response.get("result", result.stdout)

        # Extract JSON from the response text
        card_data = _extract_json(text)
        if card_data:
            return card_data

        return {"error": "Could not parse card data from response", "raw": text}

    except subprocess.TimeoutExpired:
        return {"error": "Claude CLI timed out"}
    except (json.JSONDecodeError, KeyError) as e:
        return {"error": f"Failed to parse response: {e}"}


def identify_cards_batch(image_paths: list[str]) -> list[dict]:
    """Identify multiple cards sequentially."""
    results = []
    for path in image_paths:
        result = identify_card_with_claude(path)
        results.append(result)
    return results


def _extract_json(text: str) -> dict | None:
    """Extract JSON object from a string that may contain surrounding text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON block in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None
