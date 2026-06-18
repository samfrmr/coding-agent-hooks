//! Verifies request serialization and response parsing against a local mock
//! HTTP server — no real API key or network required.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sondera_anthropic::{AnthropicClient, AnthropicConfig};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Verdict {
    flag: u8,
    label: String,
}

/// Read a full HTTP/1.1 request (headers + Content-Length body) from a stream.
fn read_request(stream: &mut std::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = stream.read(&mut chunk).unwrap();
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        let text = String::from_utf8_lossy(&buf);
        if let Some(header_end) = text.find("\r\n\r\n") {
            let content_length = text
                .lines()
                .find_map(|l| {
                    l.strip_prefix("content-length:")
                        .or_else(|| l.strip_prefix("Content-Length:"))
                })
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            let body_start = header_end + 4;
            if buf.len() >= body_start + content_length {
                break;
            }
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

#[tokio::test]
async fn complete_json_round_trips_against_mock() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel::<String>();

    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request = read_request(&mut stream);
        tx.send(request).unwrap();

        let body = r#"{"content":[{"type":"text","text":"{\"flag\":1,\"label\":\"confidential\"}"}],"stop_reason":"end_turn"}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).unwrap();
        stream.flush().unwrap();
    });

    // SAFETY: single-threaded test setup before the client reads these.
    unsafe {
        std::env::set_var("ANTHROPIC_API_KEY", "test-key-123");
    }

    let client = AnthropicClient::new(AnthropicConfig {
        model: "claude-haiku-4-5".to_string(),
        temperature: 0.0,
        base_url: format!("http://{addr}"),
    })
    .expect("client builds with key set");

    let verdict: Verdict = client
        .complete_json("system prompt", "user prompt", Duration::from_secs(5))
        .await
        .expect("mock request should succeed");

    assert_eq!(verdict.flag, 1);
    assert_eq!(verdict.label, "confidential");

    // Inspect the captured request to confirm wire shape.
    let request = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    server.join().unwrap();

    assert!(request.starts_with("POST /v1/messages"), "{request}");
    assert!(request.contains("x-api-key: test-key-123"));
    assert!(request.contains("anthropic-version: 2023-06-01"));
    assert!(request.contains("\"model\":\"claude-haiku-4-5\""));
    assert!(request.contains("\"temperature\":0.0"));
    assert!(request.contains("\"system\":\"system prompt\""));
    // Structured output config with a hardened schema.
    assert!(request.contains("\"output_config\""));
    assert!(request.contains("\"type\":\"json_schema\""));
    assert!(request.contains("\"additionalProperties\":false"));
}
