//! Strict RPC transport binding for process stdin/stdout.

#![cfg(feature = "rpc")]

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use ri_rpc::{ClientFrame, JsonlTransport, ServerFrame};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf, Stdin, Stdout};

/// One bidirectional Tokio stream backed by process stdin and stdout.
#[derive(Debug)]
pub struct Stdio {
    stdin: Stdin,
    stdout: Stdout,
}

impl Default for Stdio {
    fn default() -> Self {
        Self::new()
    }
}

impl Stdio {
    /// Bind the process standard streams.
    pub fn new() -> Self {
        Self {
            stdin: tokio::io::stdin(),
            stdout: tokio::io::stdout(),
        }
    }
}

impl AsyncRead for Stdio {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().stdin).poll_read(context, buffer)
    }
}

impl AsyncWrite for Stdio {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().stdout).poll_write(context, buffer)
    }

    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().stdout).poll_flush(context)
    }

    fn poll_shutdown(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().stdout).poll_shutdown(context)
    }

    fn is_write_vectored(&self) -> bool {
        self.stdout.is_write_vectored()
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffers: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().stdout).poll_write_vectored(context, buffers)
    }
}

/// Strict typed JSONL transport used by `ri --mode rpc`.
pub type StdioRpcTransport = JsonlTransport<Stdio, ClientFrame, ServerFrame>;

/// Construct the process RPC transport.
pub fn stdio_transport() -> StdioRpcTransport {
    JsonlTransport::new(Stdio::new())
}
