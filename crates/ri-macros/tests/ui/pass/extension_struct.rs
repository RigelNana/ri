extern crate self as ri;

pub use ri_ext as ext;

#[doc(hidden)]
pub mod __private {
    pub use async_trait::async_trait;
}

#[derive(Debug)]
#[ri_macros::extension(id = "example-struct", name = "Struct Extension")]
struct ExampleExtension;

impl ExampleExtension {
    async fn register(
        &self,
        _registrar: &mut ext::ExtensionRegistrar,
    ) -> Result<(), ext::ExtensionInitError> {
        Ok(())
    }
}

fn main() {
    let extension: std::sync::Arc<dyn ext::Extension> =
        ExampleExtension.into_extension();
    assert_eq!(extension.descriptor().id, "example-struct");
}
