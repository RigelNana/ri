//! Repair and best-effort parsing for streamed tool argument JSON.

use serde_json::{Map, Number, Value};

/// Repairs raw control characters and invalid backslash escapes inside JSON
/// strings without altering valid JSON escapes.
pub fn repair_json(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut characters = input.chars().peekable();
    let mut in_string = false;
    while let Some(character) = characters.next() {
        if !in_string {
            output.push(character);
            if character == '"' {
                in_string = true;
            }
            continue;
        }

        match character {
            '"' => {
                output.push(character);
                in_string = false;
            }
            '\\' => {
                let Some(next) = characters.peek().copied() else {
                    output.push_str("\\\\");
                    continue;
                };
                if matches!(next, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't') {
                    output.push('\\');
                    output.push(next);
                    characters.next();
                } else if next == 'u' {
                    let mut probe = characters.clone();
                    probe.next();
                    let digits = probe.by_ref().take(4).collect::<String>();
                    if digits.len() == 4 && digits.chars().all(|digit| digit.is_ascii_hexdigit()) {
                        output.push_str("\\u");
                        characters.next();
                        for _ in 0..4 {
                            if let Some(digit) = characters.next() {
                                output.push(digit);
                            }
                        }
                    } else {
                        output.push_str("\\\\");
                    }
                } else {
                    output.push_str("\\\\");
                }
            }
            '\u{0008}' => output.push_str("\\b"),
            '\u{000C}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            control if control <= '\u{001F}' => {
                use std::fmt::Write as _;
                let _ = write!(output, "\\u{:04x}", control as u32);
            }
            other => output.push(other),
        }
    }
    output
}

/// Parses complete JSON, retrying once after [`repair_json`].
///
/// # Errors
///
/// Returns the original or repaired JSON parser error when neither input can
/// be decoded as a complete JSON value.
pub fn parse_json_with_repair(input: &str) -> Result<Value, serde_json::Error> {
    serde_json::from_str(input).or_else(|original_error| {
        let repaired = repair_json(input);
        if repaired == input {
            Err(original_error)
        } else {
            serde_json::from_str(&repaired)
        }
    })
}

/// Parses incomplete streamed JSON into the useful prefix available so far.
///
/// Invalid top-level input returns an empty object. Incomplete strings,
/// objects, and arrays retain completed members and their current partial
/// string value. This keeps tool-call `arguments` object-shaped throughout a
/// stream without guessing missing scalar values.
pub fn parse_streaming_json(input: Option<&str>) -> Value {
    let Some(input) = input.filter(|input| !input.trim().is_empty()) else {
        return Value::Object(Map::new());
    };
    if let Ok(value) = parse_json_with_repair(input) {
        return value;
    }
    let repaired = repair_json(input);
    PartialParser::new(&repaired)
        .parse_value()
        .map_or_else(|| Value::Object(Map::new()), |outcome| outcome.value)
}

#[derive(Debug)]
struct ParseOutcome {
    value: Value,
    complete: bool,
}

struct PartialParser {
    characters: Vec<char>,
    cursor: usize,
}

impl PartialParser {
    fn new(input: &str) -> Self {
        Self {
            characters: input.chars().collect(),
            cursor: 0,
        }
    }

    fn parse_value(&mut self) -> Option<ParseOutcome> {
        self.skip_whitespace();
        match self.peek()? {
            '{' => self.parse_object(),
            '[' => self.parse_array(),
            '"' => {
                let (value, complete) = self.parse_string()?;
                Some(ParseOutcome {
                    value: Value::String(value),
                    complete,
                })
            }
            't' => self.parse_literal("true", Value::Bool(true)),
            'f' => self.parse_literal("false", Value::Bool(false)),
            'n' => self.parse_literal("null", Value::Null),
            '-' | '0'..='9' => self.parse_number(),
            _ => None,
        }
    }

    fn parse_object(&mut self) -> Option<ParseOutcome> {
        self.consume('{')?;
        let mut object = Map::new();
        loop {
            self.skip_whitespace();
            match self.peek() {
                Some('}') => {
                    self.cursor += 1;
                    return Some(ParseOutcome {
                        value: Value::Object(object),
                        complete: true,
                    });
                }
                Some(',') => {
                    self.cursor += 1;
                    continue;
                }
                Some('"') => {}
                None | Some(_) => {
                    return Some(ParseOutcome {
                        value: Value::Object(object),
                        complete: false,
                    });
                }
            }

            let (key, key_complete) = self.parse_string()?;
            if !key_complete {
                return Some(ParseOutcome {
                    value: Value::Object(object),
                    complete: false,
                });
            }
            self.skip_whitespace();
            if self.consume(':').is_none() {
                return Some(ParseOutcome {
                    value: Value::Object(object),
                    complete: false,
                });
            }
            let Some(value) = self.parse_value() else {
                return Some(ParseOutcome {
                    value: Value::Object(object),
                    complete: false,
                });
            };
            let value_complete = value.complete;
            object.insert(key, value.value);
            if !value_complete {
                return Some(ParseOutcome {
                    value: Value::Object(object),
                    complete: false,
                });
            }
            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.cursor += 1;
                }
                Some('}') => {}
                None | Some(_) => {
                    return Some(ParseOutcome {
                        value: Value::Object(object),
                        complete: false,
                    });
                }
            }
        }
    }

    fn parse_array(&mut self) -> Option<ParseOutcome> {
        self.consume('[')?;
        let mut array = Vec::new();
        loop {
            self.skip_whitespace();
            match self.peek() {
                Some(']') => {
                    self.cursor += 1;
                    return Some(ParseOutcome {
                        value: Value::Array(array),
                        complete: true,
                    });
                }
                None => {
                    return Some(ParseOutcome {
                        value: Value::Array(array),
                        complete: false,
                    });
                }
                Some(',') => {
                    self.cursor += 1;
                    continue;
                }
                Some(_) => {}
            }
            let Some(value) = self.parse_value() else {
                return Some(ParseOutcome {
                    value: Value::Array(array),
                    complete: false,
                });
            };
            let complete = value.complete;
            array.push(value.value);
            if !complete {
                return Some(ParseOutcome {
                    value: Value::Array(array),
                    complete: false,
                });
            }
            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.cursor += 1;
                }
                Some(']') => {}
                None | Some(_) => {
                    return Some(ParseOutcome {
                        value: Value::Array(array),
                        complete: false,
                    });
                }
            }
        }
    }

    fn parse_string(&mut self) -> Option<(String, bool)> {
        self.consume('"')?;
        let mut result = String::new();
        while let Some(character) = self.next() {
            match character {
                '"' => return Some((result, true)),
                '\\' => {
                    let Some(escape) = self.next() else {
                        return Some((result, false));
                    };
                    match escape {
                        '"' => result.push('"'),
                        '\\' => result.push('\\'),
                        '/' => result.push('/'),
                        'b' => result.push('\u{0008}'),
                        'f' => result.push('\u{000C}'),
                        'n' => result.push('\n'),
                        'r' => result.push('\r'),
                        't' => result.push('\t'),
                        'u' => {
                            let mut digits = String::new();
                            for _ in 0..4 {
                                let Some(digit) = self.next() else {
                                    return Some((result, false));
                                };
                                digits.push(digit);
                            }
                            if let Ok(code) = u16::from_str_radix(&digits, 16)
                                && let Some(decoded) = char::from_u32(u32::from(code))
                            {
                                result.push(decoded);
                            }
                        }
                        other => result.push(other),
                    }
                }
                other => result.push(other),
            }
        }
        Some((result, false))
    }

    fn parse_literal(&mut self, literal: &str, value: Value) -> Option<ParseOutcome> {
        let remaining = self.characters[self.cursor..].iter().collect::<String>();
        if remaining.starts_with(literal) {
            self.cursor += literal.chars().count();
            Some(ParseOutcome {
                value,
                complete: true,
            })
        } else {
            None
        }
    }

    fn parse_number(&mut self) -> Option<ParseOutcome> {
        let start = self.cursor;
        while self
            .peek()
            .is_some_and(|character| matches!(character, '-' | '+' | '.' | 'e' | 'E' | '0'..='9'))
        {
            self.cursor += 1;
        }
        let mut token = self.characters[start..self.cursor]
            .iter()
            .collect::<String>();
        while token.ends_with(['-', '+', '.', 'e', 'E']) {
            token.pop();
        }
        if token.is_empty() || token == "-" {
            return None;
        }
        let value = if let Ok(value) = token.parse::<i64>() {
            Value::Number(Number::from(value))
        } else if let Ok(value) = token.parse::<u64>() {
            Value::Number(Number::from(value))
        } else {
            Number::from_f64(token.parse::<f64>().ok()?).map(Value::Number)?
        };
        Some(ParseOutcome {
            value,
            complete: true,
        })
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.cursor += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.characters.get(self.cursor).copied()
    }

    fn next(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.cursor += 1;
        Some(character)
    }

    fn consume(&mut self, expected: char) -> Option<()> {
        if self.peek()? == expected {
            self.cursor += 1;
            Some(())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repairs_raw_controls_and_invalid_escapes() {
        assert_eq!(
            parse_json_with_repair("{\"path\":\"c:\\temp\\q\nnext\"}").expect("repaired")["path"],
            "c:\temp\\q\nnext"
        );
    }

    #[test]
    fn parses_nested_incomplete_objects() {
        assert_eq!(
            parse_streaming_json(Some(r#"{"path":"src/li","options":{"recursive":tr"#)),
            serde_json::json!({"path": "src/li", "options": {}})
        );
        assert_eq!(
            parse_streaming_json(Some(r#"{"a":1,"b":{"text":"hel"#)),
            serde_json::json!({"a": 1, "b": {"text": "hel"}})
        );
    }

    #[test]
    fn parses_incomplete_arrays_and_strings() {
        assert_eq!(
            parse_streaming_json(Some(r#"{"items":[1,2,"thr"#)),
            serde_json::json!({"items": [1, 2, "thr"]})
        );
    }

    #[test]
    fn malformed_or_empty_input_is_object() {
        assert_eq!(parse_streaming_json(None), serde_json::json!({}));
        assert_eq!(
            parse_streaming_json(Some("nonsense")),
            serde_json::json!({})
        );
    }
}
