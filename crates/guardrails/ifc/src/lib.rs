//! Data classification model for categorizing content sensitivity levels.
//!
//! This module provides data classification capabilities using the Anthropic API.
//! It classifies content into sensitivity levels aligned with Microsoft Purview
//! sensitivity labels: Public, General, Confidential, and Highly Confidential.
//!
//! The crate prompts a Claude model (via [`sondera_anthropic`]) to classify content
//! against sensitivity label templates following the Harmony prompt format with
//! multi-category sensitivity tiers.
//!
//! The model returns structured output with `sensitivity_category` as a [`Label`]
//! enum value (`public`, `internal`, `confidential`, `highly_confidential`),
//! enabling type-safe classification without string-based lookups.
//!
//! Requires the `ANTHROPIC_API_KEY` environment variable to be set.
//!
//! See: <https://learn.microsoft.com/en-us/purview/sensitivity-labels>

mod label;

use sondera_anthropic::{AnthropicClient, AnthropicConfig, AnthropicError};
use std::path::Path;
use std::time::Duration;
use thiserror::Error;
use tracing::instrument;

pub use label::{
    Label, LabelCategory, LabelExample, LabelTemplate, SensitivityClassification,
    SensitivityFinding, SensitivityModelResult,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during data classification.
#[derive(Debug, Error)]
pub enum DataClassificationError {
    #[error("Anthropic API error: {0}")]
    ApiError(#[from] AnthropicError),
    #[error("Failed to parse classification response: {0}")]
    ParseError(#[from] serde_json::Error),
    #[error("No label templates configured")]
    NoLabels,
    #[error("Failed to read label file: {0}")]
    IoError(String),
    #[error("Failed to parse TOML: {0}")]
    TomlError(String),
}

// ---------------------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------------------

/// Configuration for the data classification model.
#[derive(Debug, Clone)]
pub struct DataModelConfig {
    /// Model id (default: `claude-haiku-4-5`).
    pub model: String,
    /// Temperature for model inference (default: 0.0 for deterministic output).
    pub temperature: f32,
    /// API base URL (default: Anthropic, or `ANTHROPIC_BASE_URL`).
    pub base_url: String,
}

impl Default for DataModelConfig {
    fn default() -> Self {
        let defaults = AnthropicConfig::default();
        Self {
            model: defaults.model,
            temperature: defaults.temperature,
            base_url: defaults.base_url,
        }
    }
}

impl DataModelConfig {
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
}

impl From<&DataModelConfig> for AnthropicConfig {
    fn from(config: &DataModelConfig) -> Self {
        AnthropicConfig {
            model: config.model.clone(),
            temperature: config.temperature,
            base_url: config.base_url.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// DataModel
// ---------------------------------------------------------------------------

/// Data classification model using a Claude model for evaluating content
/// against sensitivity label templates with multi-category tiers.
///
/// Each [`LabelTemplate`] is evaluated independently. The model returns a
/// structured output with `sensitivity_category` as a [`Label`] enum value,
/// which is mapped to a [`SensitivityFinding`] when the content is sensitive.
///
/// # Example
///
/// ```no_run
/// use sondera_information_flow_control::{DataModel, Label, LabelTemplate};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let label = LabelTemplate::new("DATA_SENSITIVITY")
///     .description("Data sensitivity classification aligned with Microsoft Purview.")
///     .category(Label::Public, "Information that can be freely shared externally.")
///     .category(Label::HighlyConfidential, "Most sensitive data with strict access restrictions.")
///     .example("Our company was founded in 2010.", false, Label::Public)
///     .example("Employee SSN: 123-45-6789", true, Label::HighlyConfidential);
///
/// let model = DataModel::new(vec![label]);
/// let result = model.classify("Employee SSN: 123-45-6789").await?;
///
/// if result.is_sensitive() {
///     for f in &result.findings {
///         println!("{}: {}", f.label.display_name(), f.description);
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct DataModel {
    client: Option<AnthropicClient>,
    config: DataModelConfig,
    labels: Vec<LabelTemplate>,
}

impl DataModel {
    pub fn new(labels: Vec<LabelTemplate>) -> Self {
        Self::with_config(labels, DataModelConfig::default())
    }

    pub fn from_toml(path: impl AsRef<Path>) -> Result<Self, DataClassificationError> {
        let labels = LabelTemplate::load_from_toml(path)?;
        Ok(Self::new(labels))
    }

    pub fn with_config(labels: Vec<LabelTemplate>, config: DataModelConfig) -> Self {
        // Build the client eagerly; if the API key is missing it stays `None` and
        // surfaces as an error when classification is attempted (or via
        // `health_check`), keeping construction infallible.
        let client = AnthropicClient::new((&config).into()).ok();
        Self {
            client,
            config,
            labels,
        }
    }

    /// Classify content against all configured label templates.
    ///
    /// Each label is evaluated independently. A finding is recorded when
    /// `sensitive == 1` in the model's response.
    #[instrument(skip(self, content), fields(content_len = content.len()))]
    pub async fn classify(
        &self,
        content: &str,
    ) -> Result<SensitivityClassification, DataClassificationError> {
        if self.labels.is_empty() {
            return Err(DataClassificationError::NoLabels);
        }

        let mut findings = Vec::new();

        for label in &self.labels {
            let result = self
                .classify_single(label, content, Duration::from_secs(30))
                .await?;

            if result.sensitive == 1 {
                let sensitivity_label = result.sensitivity_category;
                let description = label
                    .category_definition(sensitivity_label)
                    .unwrap_or_else(|| sensitivity_label.display_name().to_string());

                findings.push(SensitivityFinding {
                    label: sensitivity_label,
                    description,
                });
            }
        }

        Ok(SensitivityClassification {
            is_public: findings.is_empty(),
            findings,
        })
    }

    /// Get the configured label templates.
    pub fn labels(&self) -> &[LabelTemplate] {
        &self.labels
    }

    /// Get the current model name.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Get the current configuration.
    pub fn config(&self) -> &DataModelConfig {
        &self.config
    }

    /// Health check to verify the Anthropic API is reachable and configured.
    ///
    /// Returns Ok(()) if the API responds within 5 seconds, Err otherwise.
    /// Use this at startup to fail fast if the API key is missing or the API is
    /// unavailable.
    pub async fn health_check(&self) -> Result<(), DataClassificationError> {
        if let Some(label) = self.labels.first() {
            self.classify_single(label, "health check", Duration::from_secs(5))
                .await?;
            Ok(())
        } else {
            Err(DataClassificationError::NoLabels)
        }
    }

    // -- private helpers ---------------------------------------------------

    async fn classify_single(
        &self,
        label: &LabelTemplate,
        content: &str,
        timeout: Duration,
    ) -> Result<SensitivityModelResult, DataClassificationError> {
        let client = self.client.as_ref().ok_or(AnthropicError::MissingApiKey)?;

        let system_prompt = label.render();
        let user_prompt = label.render_user_message(content);

        let result = client
            .complete_json::<SensitivityModelResult>(&system_prompt, &user_prompt, timeout)
            .await?;

        Ok(result)
    }
}

/// Builder for constructing a [`DataModel`] with custom configuration.
#[derive(Debug, Clone)]
pub struct DataModelBuilder {
    labels: Vec<LabelTemplate>,
    config: DataModelConfig,
}

impl DataModelBuilder {
    pub fn new() -> Self {
        Self {
            labels: Vec::new(),
            config: DataModelConfig::default(),
        }
    }

    pub fn label(mut self, label: LabelTemplate) -> Self {
        self.labels.push(label);
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.config.base_url = base_url.into();
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    pub fn build(self) -> DataModel {
        DataModel::with_config(self.labels, self.config)
    }
}

impl Default for DataModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_model_builder_custom_config() {
        let model = DataModelBuilder::new()
            .base_url("https://proxy.example.com")
            .model("claude-opus-4-8")
            .temperature(0.1)
            .label(LabelTemplate::new("L1").category(Label::Public, "Public."))
            .label(LabelTemplate::new("L2").category(Label::Public, "Public."))
            .build();

        assert_eq!(model.model(), "claude-opus-4-8");
        assert_eq!(model.config().base_url, "https://proxy.example.com");
        assert_eq!(model.labels().len(), 2);
    }

    #[test]
    fn data_model_from_toml() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../policies/ifc.toml");
        let model = DataModel::from_toml(path).unwrap();
        assert_eq!(model.labels().len(), 1);
        assert_eq!(model.model(), "claude-haiku-4-5");
    }
}
