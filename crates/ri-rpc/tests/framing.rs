//! JSONL framing properties and boundary cases.

use bytes::BytesMut;
use proptest::prelude::*;
use ri_rpc::{JsonlCodec, decode_jsonl, encode_jsonl};
use serde_json::{Value, json};
use tokio_util::codec::Decoder;

#[test]
fn lf_is_the_only_record_delimiter() {
    let text = "left\u{2028}middle\u{2029}right";
    let encoded = encode_jsonl([json!({ "text": text })]).unwrap();
    assert!(
        encoded
            .windows(3)
            .any(|bytes| bytes == "\u{2028}".as_bytes())
    );
    assert!(
        encoded
            .windows(3)
            .any(|bytes| bytes == "\u{2029}".as_bytes())
    );
    let payload = encoded
        .strip_suffix(b"\n")
        .expect("encoded JSONL must have an LF terminator");
    assert!(!payload.contains(&b'\n'));

    let decoded: Vec<Value> = decode_jsonl(&encoded).unwrap();
    assert_eq!(decoded, vec![json!({ "text": text })]);
}

#[test]
fn cr_is_stripped_only_immediately_before_lf_or_eof() {
    let input = b"{\"a\":\"x\\ry\"}\r\n{\"b\":2}\r";
    let decoded: Vec<Value> = decode_jsonl(input).unwrap();
    assert_eq!(decoded, vec![json!({"a": "x\ry"}), json!({"b": 2})]);
}

#[test]
fn decoder_handles_every_byte_as_a_chunk_boundary() {
    let expected = vec![
        json!({"text": "snowman \u{2603}"}),
        json!({"text": "line\u{2028}separator"}),
    ];
    let encoded = encode_jsonl(expected.clone()).unwrap();
    let mut codec = JsonlCodec::<Value, Value>::new();
    let mut buffer = BytesMut::new();
    let mut actual = Vec::new();

    for byte in encoded {
        buffer.extend_from_slice(&[byte]);
        while let Some(record) = codec.decode(&mut buffer).unwrap() {
            actual.push(record);
        }
    }
    assert!(codec.decode_eof(&mut buffer).unwrap().is_none());
    assert_eq!(actual, expected);
}

proptest! {
    #[test]
    fn arbitrary_json_strings_round_trip(values in prop::collection::vec(any::<String>(), 0..32)) {
        let records = values
            .iter()
            .map(|text| json!({"text": text}))
            .collect::<Vec<_>>();
        let encoded = encode_jsonl(records.clone()).unwrap();
        let decoded: Vec<Value> = decode_jsonl(&encoded).unwrap();
        prop_assert_eq!(decoded, records);
    }

    #[test]
    fn crlf_input_is_semantically_identical(values in prop::collection::vec(any::<String>(), 1..24)) {
        let records = values
            .iter()
            .map(|text| json!({"text": text}))
            .collect::<Vec<_>>();
        let encoded = encode_jsonl(records.clone()).unwrap();
        let mut crlf = Vec::with_capacity(encoded.len() + records.len());
        for byte in encoded {
            if byte == b'\n' {
                crlf.push(b'\r');
            }
            crlf.push(byte);
        }
        let decoded: Vec<Value> = decode_jsonl(&crlf).unwrap();
        prop_assert_eq!(decoded, records);
    }
}
