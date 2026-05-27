import json
import os
from pathlib import Path

import boto3
import yaml
from openai import OpenAI


DEFAULT_PROFILE = "incident_triage"
PROFILE_PATH = Path(__file__).with_name("prompt_profiles.yaml")

DEFAULT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "likely_causes": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "signals_to_check": {"type": "array", "items": {"type": "string"}},
        "automation_risk": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "summary",
        "severity",
        "likely_causes",
        "recommended_actions",
        "signals_to_check",
        "automation_risk",
    ],
}


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "content-type": "application/json",
            "access-control-allow-origin": "*",
        },
        "body": json.dumps(body),
    }


def load_profiles():
    with open(PROFILE_PATH, "r", encoding="utf-8") as profile_file:
        return yaml.safe_load(profile_file)


def parse_body(event):
    body = event.get("body", event)
    if isinstance(body, str):
        return json.loads(body or "{}")
    return body or {}


def get_openai_api_key():
    direct_key = os.getenv("OPENAI_API_KEY")
    if direct_key:
        return direct_key

    secret_arn = os.getenv("OPENAI_API_KEY_SECRET_ARN")
    if not secret_arn:
        raise RuntimeError("OPENAI_API_KEY_SECRET_ARN or OPENAI_API_KEY must be set.")

    client = boto3.client("secretsmanager")
    secret = client.get_secret_value(SecretId=secret_arn)
    return secret["SecretString"]


def build_input(payload):
    return json.dumps(
        {
            "service": payload.get("service"),
            "environment": payload.get("environment"),
            "event": payload.get("event"),
            "metrics": payload.get("metrics", {}),
            "logs": payload.get("logs", []),
            "runbook_context": payload.get("runbook_context", ""),
            "prompt_notes": payload.get("prompt_notes", ""),
        },
        indent=2,
    )


def lambda_handler(event, context):
    try:
        payload = parse_body(event)
        profiles = load_profiles()
        profile_name = payload.get("profile", DEFAULT_PROFILE)
        profile = profiles.get(profile_name)
        if profile is None:
            return response(
                400,
                {
                    "error": f"Unknown profile '{profile_name}'.",
                    "available_profiles": sorted(profiles.keys()),
                },
            )

        instructions = payload.get("instructions") or profile["instructions"]
        model = payload.get("model") or profile.get("model") or os.getenv("OPENAI_MODEL", "gpt-5.2")
        schema = payload.get("response_schema") or DEFAULT_SCHEMA

        client = OpenAI(api_key=get_openai_api_key())
        result = client.responses.create(
            model=model,
            instructions=instructions,
            input=build_input(payload),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "aiops_result",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "1200")),
        )

        return response(
            200,
            {
                "profile": profile_name,
                "model": model,
                "result": json.loads(result.output_text),
            },
        )
    except Exception as exc:
        return response(500, {"error": str(exc)})
