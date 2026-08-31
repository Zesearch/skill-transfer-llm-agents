"""Skill-update response parsers."""

from __future__ import annotations

import json


def parse_skill_update_json(raw):
    """Parse doc-memory skill update.

    Returns `{"description": str, "content": str}` or `{}`.

    Accepts either the raw model text (containing a JSON object) or a pre-parsed dict (passed straight through after a shape check).
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        desc = (raw.get("description") or "").strip()
        content = (raw.get("content") or "").strip()
        if desc and content:
            return {"description": desc, "content": content}
        return {}
    text = str(raw).strip()
    if not text or text.lower() == "none":
        return {}
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return parse_skill_update_json(obj)
    except json.JSONDecodeError:
        pass
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[brace_start:i + 1])
                        if isinstance(obj, dict):
                            return parse_skill_update_json(obj)
                    except json.JSONDecodeError:
                        break
    return {}


def _salvage_object_array(text):
    """Extract every complete JSON object from a (possibly truncated) array string, dropping a malformed or cut-off tail.

    Uses a real JSON decoder (raw_decode) per element so braces inside string values (e.g. dict literals in `implementation` source) are handled correctly, unlike naive brace-counting.
    """
    text = (text or "").strip()
    start = text.find("[")
    if start < 0:
        return []
    dec = json.JSONDecoder()
    i, n, out = start + 1, len(text), []
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            out.append(obj)
        i = end
    return out


def parse_code_skill_update_json(raw):
    """Parse code-memory skill update.

    Returns `{"functions": [{"name", "description", "implementation"}, ...]}` or `{}`.

    Accepts either the raw model text or a pre-parsed dict / list.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        functions = raw.get("functions")
        # Some models (e.g. nemotron) stringify the array argument under tool_choice="required" instead of emitting a real JSON list.
        if isinstance(functions, str):
            try:
                functions = json.loads(functions)
            except json.JSONDecodeError:
                functions = _salvage_object_array(functions)
        if isinstance(functions, list) and functions:
            norm = []
            for it in functions:
                if isinstance(it, str):
                    try:
                        it = json.loads(it)
                    except json.JSONDecodeError:
                        continue
                if isinstance(it, dict):
                    norm.append(it)
            return {"functions": norm} if norm else {}
        return {}
    if isinstance(raw, list):
        return {"functions": raw} if raw else {}
    text = str(raw).strip()
    if not text or text.lower() == "none":
        return {}
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif text.startswith("```"):
        text = text.split("```", 1)[1].split("```", 1)[0]
    text = text.strip()
    if text.startswith("["):
        try:
            functions = json.loads(text)
            if isinstance(functions, list):
                return {"functions": functions} if functions else {}
        except json.JSONDecodeError:
            salvaged = _salvage_object_array(text)
            if salvaged:
                return {"functions": salvaged}
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[brace_start:i + 1])
                        if isinstance(data, dict) and isinstance(
                                data.get("functions"), list):
                            return data if data["functions"] else {}
                    except json.JSONDecodeError:
                        break
    return {}
