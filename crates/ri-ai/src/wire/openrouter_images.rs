//! `OpenRouter` multimodal image-generation adapter.

use std::time::Duration;

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::{
    auth::ProviderHeaders,
    error::AiError,
    message::{
        AssistantImages, ImagesContext, ImagesStopReason, InputContent, TextContent, Usage,
        UsageCost, now_millis,
    },
    model::{ImageModel, ModelInput, cost_for_tokens},
    transport::{DynHttpTransport, HttpRequest, HttpResponse},
};

/// Options for a non-streaming `OpenRouter` image request.
#[derive(Clone, Debug, Default)]
pub struct OpenRouterImageOptions {
    /// Resolved bearer token.
    pub api_key: Option<String>,
    /// Caller headers; `None` suppresses a model default.
    pub headers: ProviderHeaders,
    /// Request timeout.
    pub timeout: Option<Duration>,
    /// Cooperative cancellation.
    pub cancellation: Option<CancellationToken>,
}

/// Concrete `OpenRouter` image request/response codec.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenRouterImagesAdapter;

impl OpenRouterImagesAdapter {
    /// Build the `OpenAI`-compatible Chat Completions request.
    ///
    /// # Errors
    ///
    /// Returns an authentication or validation error when the API key is
    /// absent, the endpoint is invalid, or the body cannot be serialized.
    pub fn build_request(
        &self,
        model: &ImageModel,
        context: &ImagesContext,
        options: &OpenRouterImageOptions,
    ) -> Result<HttpRequest, AiError> {
        let base = model.base_url.trim_end_matches('/');
        let endpoint = if base.ends_with("chat/completions") {
            base.to_owned()
        } else {
            format!("{base}/chat/completions")
        };
        let endpoint =
            Url::parse(&endpoint).map_err(|error| AiError::Validation(error.to_string()))?;
        let Some(api_key) = options.api_key.as_ref() else {
            return Err(AiError::Auth(format!(
                "no API key for provider {}",
                model.provider
            )));
        };
        let request_content = context
            .input
            .iter()
            .map(|input| match input {
                InputContent::Text(text) => json!({"type": "text", "text": text.text}),
                InputContent::Image(image) => json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{};base64,{}", image.mime_type, image.data)
                    }
                }),
            })
            .collect::<Vec<_>>();
        let modalities = if model.output.contains(&ModelInput::Text) {
            json!(["image", "text"])
        } else {
            json!(["image"])
        };
        let body = json!({
            "model": model.id,
            "messages": [{"role": "user", "content": request_content}],
            "stream": false,
            "modalities": modalities
        });
        let mut request = HttpRequest::json(endpoint, &body)?;
        request
            .headers
            .insert("authorization".into(), format!("Bearer {api_key}"));
        for (name, value) in &model.headers {
            request
                .headers
                .insert(name.to_ascii_lowercase(), value.clone());
        }
        for (name, value) in &options.headers {
            if let Some(value) = value {
                request
                    .headers
                    .insert(name.to_ascii_lowercase(), value.clone());
            } else {
                request.headers.remove(&name.to_ascii_lowercase());
            }
        }
        request.timeout = options.timeout;
        request.cancellation.clone_from(&options.cancellation);
        Ok(request)
    }

    /// Decode an `OpenRouter` Chat Completion into provider-neutral image output.
    ///
    /// # Errors
    ///
    /// Returns a provider or stream error when the response reports a failure
    /// or is not valid JSON.
    pub fn decode_response(
        &self,
        model: &ImageModel,
        response: &HttpResponse,
    ) -> Result<AssistantImages, AiError> {
        let value: Value = response.json()?;
        if let Some(error) = value.pointer("/error/message").and_then(Value::as_str) {
            return Err(AiError::Provider(error.to_owned()));
        }
        let mut output = AssistantImages {
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            output: Vec::new(),
            stop_reason: ImagesStopReason::Stop,
            error_message: None,
            timestamp: now_millis(),
            usage: value
                .get("usage")
                .map(|usage| parse_image_usage(model, usage)),
            response_id: value.get("id").and_then(Value::as_str).map(str::to_owned),
        };
        if let Some(text) = value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            output
                .output
                .push(InputContent::Text(TextContent::new(text)));
        }
        if let Some(images) = value
            .pointer("/choices/0/message/images")
            .and_then(Value::as_array)
        {
            for image in images {
                let Some(url) = image
                    .get("image_url")
                    .and_then(|url| url.as_str().or_else(|| url.get("url")?.as_str()))
                else {
                    continue;
                };
                if let Some((mime_type, data)) = parse_data_url(url) {
                    output
                        .output
                        .push(InputContent::Image(crate::message::ImageContent {
                            data,
                            mime_type,
                        }));
                }
            }
        }
        Ok(output)
    }

    /// Execute image generation and return a terminal provider-neutral result.
    pub async fn generate(
        &self,
        transport: DynHttpTransport,
        model: &ImageModel,
        context: &ImagesContext,
        options: &OpenRouterImageOptions,
    ) -> AssistantImages {
        match self.try_generate(transport, model, context, options).await {
            Ok(output) => output,
            Err(error) => AssistantImages {
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                output: Vec::new(),
                stop_reason: if matches!(error, AiError::Aborted)
                    || options
                        .cancellation
                        .as_ref()
                        .is_some_and(CancellationToken::is_cancelled)
                {
                    ImagesStopReason::Aborted
                } else {
                    ImagesStopReason::Error
                },
                error_message: Some(error.to_string()),
                timestamp: now_millis(),
                usage: None,
                response_id: None,
            },
        }
    }

    /// Execute image generation while preserving typed transport errors.
    ///
    /// # Errors
    ///
    /// Returns authentication, validation, transport, provider, or response
    /// decoding errors without converting them to a terminal image result.
    pub async fn try_generate(
        &self,
        transport: DynHttpTransport,
        model: &ImageModel,
        context: &ImagesContext,
        options: &OpenRouterImageOptions,
    ) -> Result<AssistantImages, AiError> {
        let request = self.build_request(model, context, options)?;
        let response = transport.execute(request).await?;
        self.decode_response(model, &response)
    }
}

fn parse_data_url(url: &str) -> Option<(String, String)> {
    let value = url.strip_prefix("data:")?;
    let (mime, encoded) = value.split_once(";base64,")?;
    if mime.is_empty() || encoded.is_empty() {
        return None;
    }
    Some((mime.to_owned(), encoded.to_owned()))
}

fn parse_image_usage(model: &ImageModel, value: &Value) -> Usage {
    let prompt = value
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reported_cached = value
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_write = value
        .pointer("/prompt_tokens_details/cache_write_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read = reported_cached.saturating_sub(cache_write);
    let input = prompt
        .saturating_sub(cache_read)
        .saturating_sub(cache_write);
    let output = value
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mut cost = UsageCost {
        input: cost_for_tokens(model.cost.rates.input, input),
        output: cost_for_tokens(model.cost.rates.output, output),
        cache_read: cost_for_tokens(model.cost.rates.cache_read, cache_read),
        cache_write: cost_for_tokens(model.cost.rates.cache_write, cache_write),
        total: 0.0,
    };
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write;
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: input
            .saturating_add(output)
            .saturating_add(cache_read)
            .saturating_add(cache_write),
        cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelCost, ModelCostRates, ModelInput};
    use crate::transport::HttpHeaders;
    use bytes::Bytes;

    fn model() -> ImageModel {
        ImageModel {
            id: "image/model".into(),
            name: "Image".into(),
            api: "openrouter-images".into(),
            provider: "openrouter".into(),
            base_url: "https://openrouter.ai/api/v1".into(),
            input: vec![ModelInput::Text, ModelInput::Image],
            output: vec![ModelInput::Image, ModelInput::Text],
            cost: ModelCost {
                rates: ModelCostRates {
                    input: 1.0,
                    output: 2.0,
                    cache_read: 0.5,
                    cache_write: 1.25,
                },
                tiers: Vec::new(),
            },
            headers: HttpHeaders::new(),
        }
    }

    #[test]
    fn decoder_extracts_text_and_data_urls() {
        let response = HttpResponse {
            status: 200,
            headers: HttpHeaders::new(),
            body: Bytes::from(
                json!({
                    "id": "generation",
                    "choices": [{
                        "message": {
                            "content": "caption",
                            "images": [
                                {"image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                                {"image_url": "https://example.test/not-inline"}
                            ]
                        }
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 1}
                    }
                })
                .to_string(),
            ),
        };
        let output = OpenRouterImagesAdapter
            .decode_response(&model(), &response)
            .expect("response");
        assert_eq!(output.response_id.as_deref(), Some("generation"));
        assert_eq!(output.output.len(), 2);
        assert!(matches!(
            &output.output[1],
            InputContent::Image(image)
                if image.mime_type == "image/png" && image.data == "aGVsbG8="
        ));
        let usage = output.usage.expect("usage");
        assert_eq!(usage.input, 7);
        assert_eq!(usage.cache_read, 2);
        assert_eq!(usage.cache_write, 1);
    }

    #[test]
    fn malformed_data_urls_are_ignored() {
        assert_eq!(parse_data_url("data:image/png,raw"), None);
        assert_eq!(
            parse_data_url("data:image/png;base64,abc"),
            Some(("image/png".into(), "abc".into()))
        );
    }
}
