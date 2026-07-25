//! Stable JSON projection of high-level SDK events.

use std::fmt;

use async_trait::async_trait;
use ri_sdk::{HarnessEvent, HarnessObserver};
use serde_json::{Value, json};
use tokio::sync::broadcast;

pub(super) struct EventObserver {
    sender: broadcast::Sender<Value>,
    harness_sender: broadcast::Sender<HarnessEvent>,
}

impl EventObserver {
    pub(super) fn new(
        sender: broadcast::Sender<Value>,
        harness_sender: broadcast::Sender<HarnessEvent>,
    ) -> Self {
        Self {
            sender,
            harness_sender,
        }
    }
}

impl fmt::Debug for EventObserver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EventObserver")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl HarnessObserver for EventObserver {
    async fn on_event(&self, event: &HarnessEvent) -> ri_harness::Result<()> {
        // Frontend streams are optional observers. With no subscriber,
        // `broadcast::Sender::send` correctly reports that nothing consumed the
        // event; it must not fail the authoritative harness operation.
        let _ = self.sender.send(event_value(event));
        let _ = self.harness_sender.send(event.clone());
        Ok(())
    }
}

pub(super) fn event_value(event: &HarnessEvent) -> Value {
    match event {
        HarnessEvent::ResourceExpanded { resource, text } => {
            let (kind, name, source) = match resource {
                ri_harness::ExpandedResource::Skill { name, source } => ("skill", name, source),
                ri_harness::ExpandedResource::PromptTemplate { name, source } => {
                    ("prompt_template", name, source)
                }
            };
            json!({
                "type": "resource_expanded",
                "kind": kind,
                "name": name,
                "source": source,
                "text": text,
            })
        }
        HarnessEvent::PromptAccepted { operation } => {
            json!({"type": "prompt_accepted", "operation": operation})
        }
        HarnessEvent::QueueUpdated(lengths) => json!({
            "type": "queue_updated",
            "steer": lengths.steer,
            "followUp": lengths.follow_up,
            "nextTurn": lengths.next_turn,
        }),
        HarnessEvent::MessagePersisted { entry_id, role } => json!({
            "type": "message_persisted",
            "entryId": entry_id,
            "role": role,
        }),
        HarnessEvent::SavePoint {
            operation,
            had_pending_writes,
        } => json!({
            "type": "save_point",
            "operation": operation,
            "hadPendingWrites": had_pending_writes,
        }),
        HarnessEvent::RetryScheduled {
            operation,
            attempt,
            max_attempts,
            delay,
            error,
        } => match operation {
            ri_harness::RetryOperation::Agent => json!({
                "type": "auto_retry_start",
                "attempt": attempt,
                "maxAttempts": max_attempts,
                "delayMs": duration_millis(*delay),
                "errorMessage": error,
            }),
            ri_harness::RetryOperation::Compaction
            | ri_harness::RetryOperation::TurnPrefix
            | ri_harness::RetryOperation::BranchSummary => json!({
                "type": "summarization_retry_scheduled",
                "attempt": attempt,
                "maxAttempts": max_attempts,
                "delayMs": duration_millis(*delay),
                "errorMessage": error,
            }),
        },
        HarnessEvent::RetryAttemptStarted { kind, reason } => json!({
            "type": "summarization_retry_attempt_start",
            "source": match kind {
                ri_harness::SummaryKind::Branch => "branchSummary",
                ri_harness::SummaryKind::Compaction | ri_harness::SummaryKind::TurnPrefix => "compaction",
            },
            "reason": reason.map(ri_harness::CompactionReason::as_str),
        }),
        HarnessEvent::RetryFinished {
            operation,
            success,
            attempt,
            final_error,
        } => match operation {
            ri_harness::RetryOperation::Agent => json!({
                "type": "auto_retry_end",
                "success": success,
                "attempt": attempt,
                "finalError": final_error,
            }),
            ri_harness::RetryOperation::Compaction
            | ri_harness::RetryOperation::TurnPrefix
            | ri_harness::RetryOperation::BranchSummary => json!({
                "type": "summarization_retry_finished",
            }),
        },
        HarnessEvent::CompactionStarted { reason } => json!({
            "type": "compaction_start",
            "reason": reason.as_str(),
        }),
        HarnessEvent::CompactionFinished {
            reason,
            result,
            aborted,
            will_retry,
            error_message,
        } => json!({
            "type": "compaction_end",
            "reason": reason.as_str(),
            "result": result.as_ref().map(|result| json!({
                "summary": result.summary,
                "firstKeptEntryId": result.first_kept_entry_id,
                "tokensBefore": result.tokens_before,
                "estimatedTokensAfter": result.estimated_tokens_after,
                "usage": result.usage,
                "details": result.details,
            })),
            "aborted": aborted,
            "willRetry": will_retry,
            "errorMessage": error_message,
        }),
        HarnessEvent::BranchNavigated {
            old_leaf,
            new_leaf,
            summary_entry,
        } => json!({
            "type": "branch_navigated",
            "oldLeaf": old_leaf,
            "newLeaf": new_leaf,
            "summaryEntry": summary_entry,
        }),
        HarnessEvent::SessionReplacing { old_session_id } => json!({
            "type": "session_replacing",
            "oldSessionId": old_session_id,
        }),
        HarnessEvent::SessionReplaced {
            session_id,
            generation,
        } => json!({
            "type": "session_replaced",
            "sessionId": session_id,
            "generation": generation,
        }),
        HarnessEvent::Settled {
            operation,
            next_turn,
        } => json!({
            "type": "settled",
            "operation": operation,
            "nextTurn": next_turn,
        }),
    }
}

pub(super) fn duration_millis(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_projection_has_stable_type() {
        let event = HarnessEvent::Settled {
            operation: 7,
            next_turn: 2,
        };
        assert_eq!(
            event_value(&event),
            json!({"type": "settled", "operation": 7, "nextTurn": 2})
        );
    }
}
