extern crate self as ri;

pub use ri_ext as ext;

#[doc(hidden)]
pub mod __private {
    pub use async_trait::async_trait;
}

#[ri_macros::extension(
    id = "example-function",
    name = "Function Extension",
    version = "1.0.0"
)]
async fn register(
    _registrar: &mut ext::ExtensionRegistrar,
) -> Result<(), ext::ExtensionInitError> {
    Ok(())
}

fn main() {
    let extension: std::sync::Arc<dyn ext::Extension> = register_extension();
    assert_eq!(extension.descriptor().id, "example-function");
}
