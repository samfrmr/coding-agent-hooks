//! Minimal Anthropic Messages API client for the LLM-based guardrail classifiers.
//!
//! The guardrail crates ([`sondera-policy`] and [`sondera-information-flow-control`])
//! classify content by prompting a model and parsing a small structured JSON reply.
//! This crate owns the shared HTTP plumbing: it calls
//! `POST {base_url}/v1/messages`, constrains the reply with the structured-output
//! `output_config.format` so the model returns schema-valid JSON, and deserializes
//! the first text block into a caller-provided type.
//!
//! Rust has no official Anthropic SDK, so this uses raw HTTP via [`reqwest`].
//!
//! The API key is read from the `ANTHROPIC_API_KEY` environment variable, which is
//! populated by the hook binaries from `~/.sondera/env`. The base URL defaults to
//! `https://api.anthropic.com` and can be overridden with `ANTHROPIC_BASE_URL`
//! (useful for proxies or a mock server in tests).
//!
//! [`sondera-policy`]: https://docs.rs/sondera-policy
//! [`sondera-information-flow-control`]: https://docs.rs/sondera-information-flow-control

use std::time::Duration;

use schemars::JsonSchema;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use thiserror::Error;

/// API version sent with every request (see Anthropic API docs).
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default model used by the classifiers. Haiku is the cheapest/fastest model and,
/// unlike Opus 4.8/4.7, still accepts `temperature` so deterministic (`0.0`)
/// classification is preserved.
pub const DEFAULT_MODEL: &str = "claude-haiku-4-5";

/// Default API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Output cap for a classification reply. The structured output is a tiny JSON
/// object, so this is generous headroom.
const MAX_TOKENS: u32 = 1024;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur talking to the Anthropic API.
#[derive(Debug, Error)]
pub enum AnthropicError {
    #[error("ANTHROPIC_API_KEY is not set")]
    MissingApiKey,
    #[error("HTTP transport error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Anthropic API error ({status}): {message}")]
    Api { status: u16, message: String },
    #[error("Request timed out")]
    Timeout,
    #[error("Model refused the request")]
    Refusal,
    #[error("Response contained no text content block")]
    NoContent,
    #[error("Failed to parse model response: {0}")]
    Parse(#[from] serde_json::Error),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Anthropic client.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// Model id (default: [`DEFAULT_MODEL`]).
    pub model: String,
    /// Sampling temperature (default: `0.0` for deterministic output).
    pub temperature: f32,
    /// API base URL (default: [`DEFAULT_BASE_URL`], or `ANTHROPIC_BASE_URL`).
    pub base_url: String,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            temperature: 0.0,
            base_url: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// A minimal Anthropic Messages API client.
pub struct AnthropicClient {
    http: reqwest::Client,
    api_key: String,
    config: AnthropicConfig,
}

impl AnthropicClient {
    /// Build a client, reading the API key from `ANTHROPIC_API_KEY`.
    ///
    /// Returns [`AnthropicError::MissingApiKey`] if the variable is unset or empty.
    pub fn new(config: AnthropicConfig) -> Result<Self, AnthropicError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .ok_or(AnthropicError::MissingApiKey)?;
        Ok(Self {
            http: reqwest::Client::new(),
            api_key,
            config,
        })
    }

    /// The configured model id.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Prompt the model with a system + user message and deserialize the reply
    /// into `T`. The reply is constrained to `T`'s JSON schema via structured
    /// outputs, so the returned text is schema-valid JSON.
    pub async fn complete_json<T>(
        &self,
        system: &str,
        user: &str,
        timeout: Duration,
    ) -> Result<T, AnthropicError>
    where
        T: DeserializeOwned + JsonSchema,
    {
        let schema = harden_schema(serde_json::to_value(schemars::schema_for!(T))?);

        let body = json!({
            "model": self.config.model,
            "max_tokens": MAX_TOKENS,
            "temperature": self.config.temperature,
            "system": system,
            "messages": [{ "role": "user", "content": user }],
            "output_config": {
                "format": { "type": "json_schema", "schema": schema }
            },
        });

        let url = format!("{}/v1/messages", self.config.base_url.trim_end_matches('/'));

        let response = self
            .http
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    AnthropicError::Timeout
                } else {
                    AnthropicError::Http(e)
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let message = serde_json::from_str::<ApiErrorEnvelope>(&text)
                .map(|e| e.error.message)
                .unwrap_or(text);
            return Err(AnthropicError::Api {
                status: status.as_u16(),
                message,
            });
        }

        let message: MessagesResponse = response.json().await?;
        if message.stop_reason.as_deref() == Some("refusal") {
            return Err(AnthropicError::Refusal);
        }

        let text = message
            .content
            .into_iter()
            .find_map(|block| match block {
                ContentBlock::Text { text } => Some(text),
                ContentBlock::Other => None,
            })
            .ok_or(AnthropicError::NoContent)?;

        Ok(serde_json::from_str(&text)?)
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct MessagesResponse {
    #[serde(default)]
    content: Vec<ContentBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct ApiErrorEnvelope {
    error: ApiErrorDetail,
}

#[derive(Deserialize)]
struct ApiErrorDetail {
    message: String,
}

// ---------------------------------------------------------------------------
// Schema hardening
// ---------------------------------------------------------------------------

/// Adjust a `schemars`-generated schema to satisfy Anthropic's structured-output
/// rules: every object schema must set `additionalProperties: false`, and the
/// root must not carry the `$schema` / `title` metadata keys.
fn harden_schema(mut schema: Value) -> Value {
    inline_refs(&mut schema);
    collapse_const_unions(&mut schema);
    set_additional_properties(&mut schema);
    strip_unsupported_numeric_keywords(&mut schema);
    if let Value::Object(map) = &mut schema {
        map.remove("$schema");
        map.remove("title");
    }
    schema
}

/// Inline every `$ref: "#/$defs/Name"` by substituting the referenced subschema,
/// then drop the now-unused `$defs`. `schemars` factors named types (such as the
/// `Label` enum) into `$defs` and references them, but the Anthropic
/// structured-output API expects a single self-contained schema. The result
/// types here are acyclic, so a straightforward recursive substitution suffices.
fn inline_refs(schema: &mut Value) {
    let defs = schema
        .get("$defs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    substitute_refs(schema, &defs);
    if let Value::Object(map) = schema {
        map.remove("$defs");
    }
}

fn substitute_refs(value: &mut Value, defs: &serde_json::Map<String, Value>) {
    match value {
        Value::Object(map) => {
            if let Some(name) = map
                .get("$ref")
                .and_then(Value::as_str)
                .and_then(|r| r.strip_prefix("#/$defs/"))
            {
                if let Some(target) = defs.get(name) {
                    let mut resolved = target.clone();
                    substitute_refs(&mut resolved, defs);
                    *value = resolved;
                    return;
                }
            }
            for child in map.values_mut() {
                substitute_refs(child, defs);
            }
        }
        Value::Array(items) => {
            for child in items {
                substitute_refs(child, defs);
            }
        }
        _ => {}
    }
}

/// Recursively collapse `oneOf`/`anyOf` unions of string `const`s into a single
/// `{"type":"string","enum":[...]}` node. `schemars` renders a Rust enum whose
/// variants carry doc comments as `oneOf: [{const: "a", description: ...}, ...]`,
/// but the Anthropic structured-output API rejects `oneOf`/`anyOf`. Collapsing to
/// a flat string `enum` preserves the allowed values (dropping per-variant
/// descriptions, which the API does not use for constraint anyway).
fn collapse_const_unions(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for key in ["oneOf", "anyOf"] {
                let consts = map.get(key).and_then(string_consts);
                if let Some(values) = consts {
                    map.remove("oneOf");
                    map.remove("anyOf");
                    map.insert("type".into(), Value::String("string".into()));
                    map.insert("enum".into(), Value::Array(values));
                    break;
                }
            }
            for child in map.values_mut() {
                collapse_const_unions(child);
            }
        }
        Value::Array(items) => {
            for child in items {
                collapse_const_unions(child);
            }
        }
        _ => {}
    }
}

/// If `value` is an array in which every element is an object with a string
/// `const`, return the list of those const values; otherwise `None`.
fn string_consts(value: &Value) -> Option<Vec<Value>> {
    let arr = value.as_array()?;
    if arr.is_empty() {
        return None;
    }
    arr.iter()
        .map(|v| match v.get("const") {
            Some(c @ Value::String(_)) => Some(c.clone()),
            _ => None,
        })
        .collect()
}

/// Recursively remove numeric keywords that the Anthropic structured-output API
/// rejects. `schemars` emits `minimum`/`maximum` (and a `uint8`-style `format`)
/// for bounded integer types such as `u8`, but the API returns a 400 for
/// `minimum`/`maximum` on `integer`/`number` nodes. The paired `format` is
/// dropped as well so we don't trade one 400 for the next.
fn strip_unsupported_numeric_keywords(value: &mut Value) {
    match value {
        Value::Object(map) => {
            let is_numeric = matches!(
                map.get("type").and_then(Value::as_str),
                Some("integer") | Some("number")
            );
            if is_numeric {
                map.remove("minimum");
                map.remove("maximum");
                map.remove("exclusiveMinimum");
                map.remove("exclusiveMaximum");
                map.remove("format");
            }
            for child in map.values_mut() {
                strip_unsupported_numeric_keywords(child);
            }
        }
        Value::Array(items) => {
            for child in items {
                strip_unsupported_numeric_keywords(child);
            }
        }
        _ => {}
    }
}

/// Recursively add `additionalProperties: false` to every object-typed schema
/// node that declares `properties`.
fn set_additional_properties(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for child in map.values_mut() {
                set_additional_properties(child);
            }
            let is_object = map.get("type").and_then(Value::as_str) == Some("object");
            if is_object && map.contains_key("properties") {
                map.entry("additionalProperties")
                    .or_insert(Value::Bool(false));
            }
        }
        Value::Array(items) => {
            for child in items {
                set_additional_properties(child);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, JsonSchema)]
    struct Sample {
        flag: u8,
        label: String,
        kind: Kind,
    }

    #[derive(Serialize, Deserialize, JsonSchema)]
    #[serde(rename_all = "snake_case")]
    enum Kind {
        /// First variant — doc comment forces schemars into a `oneOf` of consts.
        Alpha,
        /// Second variant.
        Beta,
    }

    #[test]
    fn config_defaults_to_haiku() {
        let cfg = AnthropicConfig::default();
        assert_eq!(cfg.model, "claude-haiku-4-5");
        assert_eq!(cfg.temperature, 0.0);
    }

    #[test]
    fn hardened_schema_marks_object_closed_and_strips_meta() {
        let schema = harden_schema(serde_json::to_value(schemars::schema_for!(Sample)).unwrap());
        let map = schema.as_object().unwrap();
        assert!(!map.contains_key("$schema"));
        assert!(!map.contains_key("title"));
        assert_eq!(
            map.get("additionalProperties"),
            Some(&Value::Bool(false)),
            "root object schema must be closed"
        );
        assert!(map.contains_key("properties"));
    }

    #[test]
    fn hardened_schema_strips_numeric_bounds_for_integers() {
        // `u8` makes schemars emit minimum/maximum/format, which the Anthropic
        // structured-output API rejects with a 400. They must be stripped.
        let schema = harden_schema(serde_json::to_value(schemars::schema_for!(Sample)).unwrap());
        let flag = schema
            .get("properties")
            .and_then(|p| p.get("flag"))
            .and_then(Value::as_object)
            .expect("flag property schema");
        assert_eq!(flag.get("type").and_then(Value::as_str), Some("integer"));
        assert!(!flag.contains_key("minimum"), "minimum must be stripped");
        assert!(!flag.contains_key("maximum"), "maximum must be stripped");
        assert!(!flag.contains_key("format"), "format must be stripped");
    }

    #[test]
    fn hardened_schema_inlines_refs_and_collapses_enum_to_string_enum() {
        // An enum with doc-commented variants is factored into `$defs` and
        // referenced via `$ref`, with the variants encoded as `oneOf: [{const}]`.
        // The API rejects both `$ref`/`$defs` and `oneOf`, so after hardening the
        // `kind` property must be an inlined, flat string `enum`.
        let schema = harden_schema(serde_json::to_value(schemars::schema_for!(Sample)).unwrap());
        assert!(
            schema.get("$defs").is_none(),
            "$defs must be removed after inlining"
        );
        let kind = schema
            .get("properties")
            .and_then(|p| p.get("kind"))
            .and_then(Value::as_object)
            .expect("kind property schema");
        assert!(!kind.contains_key("$ref"), "$ref must be inlined");
        assert!(!kind.contains_key("oneOf"), "oneOf must be collapsed");
        assert!(!kind.contains_key("anyOf"), "anyOf must be collapsed");
        assert_eq!(kind.get("type").and_then(Value::as_str), Some("string"));
        assert_eq!(
            kind.get("enum").and_then(Value::as_array),
            Some(&vec![Value::from("alpha"), Value::from("beta")]),
            "variants must be preserved as a string enum"
        );
    }
}
