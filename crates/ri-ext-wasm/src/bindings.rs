//! Generated Component Model bindings for the versioned extension world.

wasmtime::component::bindgen!({
    path: "../../wit/ri-extension.wit",
    world: "extension",
    imports: { default: async | trappable },
    exports: { default: async },
    additional_derives: [serde::Serialize],
    with: {
        "ri:extension/filesystem.filesystem": crate::FilesystemResource,
        "ri:extension/network.network": crate::NetworkResource,
        "ri:extension/process.process": crate::ProcessResource,
        "ri:extension/ui.ui": crate::UiResource,
        "ri:extension/session.session": crate::SessionResource,
        "ri:extension/provider.provider": crate::ProviderResource,
    },
});

#[cfg(test)]
mod tests {
    use std::path::Path;
    use wit_parser::Resolve;

    #[test]
    fn versioned_extension_world_parses() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../wit/ri-extension.wit");
        let mut resolve = Resolve::default();
        let package = resolve
            .push_file(path)
            .expect("ri extension WIT must parse");
        let world = resolve
            .select_world(&[package], Some("extension"))
            .expect("extension world must be selectable");
        assert_eq!(resolve.worlds[world].name, "extension");
        let version = resolve.packages[package]
            .name
            .version
            .as_ref()
            .expect("package must be versioned");
        assert_eq!(version.to_string(), crate::ABI_VERSION);
    }
}
