//! Extensible application-message projection.

use std::fmt::Debug;

use ri_ai::Message;

/// An application message that can participate in an agent transcript.
///
/// Implementations may wrap the three provider-facing [`Message`] variants and
/// add arbitrary UI or session-only variants. [`Self::project`] is evaluated at
/// every provider boundary, after the optional context transform.
pub trait AgentMessage: Clone + Debug + Send + Sync + 'static {
    /// Returns the embedded provider-facing message, when this value directly
    /// represents one.
    ///
    /// The loop uses this method only for continuation validation. Custom
    /// variants may return `None` even when [`Self::project`] creates a message.
    fn as_llm_message(&self) -> Option<&Message>;

    /// Wraps a provider-facing message for insertion into the transcript.
    fn from_llm_message(message: Message) -> Self;

    /// Projects this value into provider context.
    ///
    /// Returning `None` filters an application-only value from the request.
    fn project(&self) -> Option<Message> {
        self.as_llm_message().cloned()
    }
}

impl AgentMessage for Message {
    fn as_llm_message(&self) -> Option<&Message> {
        Some(self)
    }

    fn from_llm_message(message: Message) -> Self {
        message
    }
}

/// Input accepted by [`crate::Agent::prompt`].
#[derive(Clone, Debug, PartialEq)]
pub enum Prompt<M> {
    /// A single application message.
    One(M),
    /// An ordered batch of application messages.
    Many(Vec<M>),
}

impl<M> Prompt<M> {
    /// Converts the prompt into its ordered message list.
    pub fn into_messages(self) -> Vec<M> {
        match self {
            Self::One(message) => vec![message],
            Self::Many(messages) => messages,
        }
    }
}

impl<M> From<M> for Prompt<M> {
    fn from(message: M) -> Self {
        Self::One(message)
    }
}

impl<M> From<Vec<M>> for Prompt<M> {
    fn from(messages: Vec<M>) -> Self {
        Self::Many(messages)
    }
}

impl From<String> for Prompt<Message> {
    fn from(text: String) -> Self {
        Self::One(Message::User(ri_ai::UserMessage::new(text)))
    }
}

impl From<&str> for Prompt<Message> {
    fn from(text: &str) -> Self {
        Self::from(text.to_owned())
    }
}

/// The provider-facing message type used by the default agent specialization.
pub type StandardAgentMessage = Message;
