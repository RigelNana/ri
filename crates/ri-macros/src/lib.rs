//! Procedural adapters for native Ri tools and extensions.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{
    Attribute, Expr, ExprLit, FnArg, Item, ItemFn, ItemStruct, Lit, Meta, MetaNameValue,
    ReturnType, Token, Type, parse_macro_input,
};

#[derive(Default)]
struct Arguments {
    id: Option<String>,
    name: Option<String>,
    label: Option<String>,
    description: Option<String>,
    version: Option<String>,
    factory: Option<String>,
    execution: Option<String>,
}

/// Turns a typed function into an erased `ri_agent::Tool` factory.
///
/// Both synchronous and asynchronous functions are supported, with an optional
/// leading `ToolCallContext` followed by the typed input. A zero-argument
/// function uses `()` as its generated schema. Attribute keys are `name`,
/// `label`, `description`, `factory`, and
/// `execution = "parallel" | "sequential"`.
#[proc_macro_attribute]
pub fn tool(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let arguments = match parse_arguments(attributes) {
        Ok(arguments) => arguments,
        Err(error) => return error.into_compile_error().into(),
    };
    let function = parse_macro_input!(item as ItemFn);
    expand_tool(arguments, &function)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Implements and type-erases an `ri_ext::Extension`.
///
/// On an async registration function, this generates a zero-sized extension
/// wrapper and `<function>_extension()` factory. On a struct, it implements the
/// trait by delegating to an inherent async
/// `register(&self, &mut ExtensionRegistrar)` method and adds
/// `into_extension(self)`.
#[proc_macro_attribute]
pub fn extension(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let arguments = match parse_arguments(attributes) {
        Ok(arguments) => arguments,
        Err(error) => return error.into_compile_error().into(),
    };
    let item = parse_macro_input!(item as Item);
    let expanded = match item {
        Item::Fn(function) => expand_extension_function(arguments, &function),
        Item::Struct(structure) => expand_extension_struct(arguments, &structure),
        other => Err(syn::Error::new_spanned(
            other,
            "#[ri::extension] supports an async function or a struct",
        )),
    };
    expanded
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn parse_arguments(input: TokenStream) -> syn::Result<Arguments> {
    let values = Punctuated::<MetaNameValue, Token![,]>::parse_terminated.parse(input)?;
    let mut arguments = Arguments::default();
    for value in values {
        let Some(identifier) = value.path.get_ident() else {
            return Err(syn::Error::new_spanned(
                value.path,
                "attribute key must be an identifier",
            ));
        };
        let Expr::Lit(ExprLit {
            lit: Lit::Str(value),
            ..
        }) = value.value
        else {
            return Err(syn::Error::new_spanned(
                value.value,
                "attribute values must be string literals",
            ));
        };
        let slot = match identifier.to_string().as_str() {
            "id" => &mut arguments.id,
            "name" => &mut arguments.name,
            "label" => &mut arguments.label,
            "description" => &mut arguments.description,
            "version" => &mut arguments.version,
            "factory" => &mut arguments.factory,
            "execution" => &mut arguments.execution,
            _ => {
                return Err(syn::Error::new_spanned(
                    identifier,
                    "unsupported ri macro attribute key",
                ));
            }
        };
        if slot.replace(value.value()).is_some() {
            return Err(syn::Error::new_spanned(
                identifier,
                "duplicate ri macro attribute key",
            ));
        }
    }
    Ok(arguments)
}

fn expand_tool(arguments: Arguments, function: &ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    require_plain_function(function)?;
    if arguments.id.is_some() || arguments.version.is_some() {
        return Err(syn::Error::new_spanned(
            &function.sig.ident,
            "tool attributes do not support id or version",
        ));
    }
    let function_name = &function.sig.ident;
    let visibility = &function.vis;
    let tool_name = arguments
        .name
        .unwrap_or_else(|| unraw(&function_name.to_string()));
    let label = arguments.label.unwrap_or_else(|| title_case(&tool_name));
    let description = arguments
        .description
        .or_else(|| doc_text(&function.attrs))
        .unwrap_or_else(|| format!("Runs the {tool_name} tool."));
    let factory = factory_ident(
        arguments.factory.as_deref(),
        &format!("{}_tool", unraw(&function_name.to_string())),
        function_name,
    )?;

    let typed = function
        .sig
        .inputs
        .iter()
        .map(argument_type)
        .collect::<syn::Result<Vec<_>>>()?;
    let (parameter_type, call) = match typed.as_slice() {
        [] => (quote! { () }, quote! { #function_name() }),
        [parameter] => (
            quote! { #parameter },
            quote! { #function_name(__ri_arguments) },
        ),
        [_context, parameter] => (
            quote! { #parameter },
            quote! { #function_name(__ri_context, __ri_arguments) },
        ),
        _ => {
            return Err(syn::Error::new_spanned(
                &function.sig.inputs,
                "tool functions accept at most ToolCallContext and one typed argument",
            ));
        }
    };
    let invocation = if function.sig.asyncness.is_some() {
        quote! { #call.await }
    } else {
        call
    };
    if matches!(function.sig.output, ReturnType::Default) {
        return Err(syn::Error::new_spanned(
            &function.sig,
            "tool functions must return Result<ToolResult, ToolError>",
        ));
    }
    let execution = match arguments.execution.as_deref() {
        None => quote! {},
        Some("parallel") => quote! {
            let __ri_tool = __ri_tool.with_execution_mode(
                ::ri::agent::ToolExecutionMode::Parallel
            );
        },
        Some("sequential") => quote! {
            let __ri_tool = __ri_tool.with_execution_mode(
                ::ri::agent::ToolExecutionMode::Sequential
            );
        },
        Some(other) => {
            return Err(syn::Error::new_spanned(
                &function.sig.ident,
                format!("unsupported tool execution mode {other:?}"),
            ));
        }
    };

    Ok(quote! {
        #function

        #[doc = "Creates the generated schema-checked, type-erased Ri tool."]
        #visibility fn #factory(
        ) -> ::std::result::Result<
            ::std::sync::Arc<dyn ::ri::agent::Tool>,
            ::ri::agent::ToolError,
        > {
            let __ri_tool = ::ri::agent::FnTool::typed::<#parameter_type, _, _>(
                #tool_name,
                #label,
                #description,
                move |__ri_context, __ri_arguments| async move {
                    #invocation
                },
            )?;
            #execution
            ::std::result::Result::Ok(::std::sync::Arc::new(__ri_tool))
        }
    })
}

fn expand_extension_function(
    arguments: Arguments,
    function: &ItemFn,
) -> syn::Result<proc_macro2::TokenStream> {
    require_async(function, "extension")?;
    require_plain_function(function)?;
    if arguments.label.is_some() || arguments.description.is_some() || arguments.execution.is_some()
    {
        return Err(syn::Error::new_spanned(
            &function.sig.ident,
            "extension attributes do not support label, description, or execution",
        ));
    }
    if function.sig.inputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            &function.sig.inputs,
            "extension registration functions take one &mut ExtensionRegistrar argument",
        ));
    }
    let function_name = &function.sig.ident;
    let visibility = &function.vis;
    let default_id = unraw(&function_name.to_string()).replace('_', "-");
    let id = arguments.id.unwrap_or(default_id);
    let name = arguments
        .name
        .unwrap_or_else(|| title_case(&id.replace('-', "_")));
    let version = option_string(arguments.version);
    let factory = factory_ident(
        arguments.factory.as_deref(),
        &format!("{}_extension", unraw(&function_name.to_string())),
        function_name,
    )?;
    let wrapper = format_ident!(
        "{}Extension",
        pascal_case(&unraw(&function_name.to_string()))
    );

    Ok(quote! {
        #function

        #[doc = "Generated native extension wrapper."]
        #[derive(Debug, Default)]
        #visibility struct #wrapper;

        #[::ri::__private::async_trait]
        impl ::ri::ext::Extension for #wrapper {
            fn descriptor(&self) -> ::ri::ext::ExtensionDescriptor {
                ::ri::ext::ExtensionDescriptor {
                    id: #id.to_owned(),
                    name: #name.to_owned(),
                    version: #version,
                    source: ::ri::ext::SourceInfo::inline(#id),
                }
            }

            async fn register(
                &self,
                __ri_registrar: &mut ::ri::ext::ExtensionRegistrar,
            ) -> ::std::result::Result<(), ::ri::ext::ExtensionInitError> {
                #function_name(__ri_registrar).await
            }
        }

        #[doc = "Creates the generated type-erased native Ri extension."]
        #visibility fn #factory() -> ::std::sync::Arc<dyn ::ri::ext::Extension> {
            ::std::sync::Arc::new(#wrapper)
        }
    })
}

fn expand_extension_struct(
    arguments: Arguments,
    structure: &ItemStruct,
) -> syn::Result<proc_macro2::TokenStream> {
    if arguments.factory.is_some()
        || arguments.label.is_some()
        || arguments.description.is_some()
        || arguments.execution.is_some()
    {
        return Err(syn::Error::new_spanned(
            &structure.ident,
            "struct extensions do not support factory, label, description, or execution",
        ));
    }
    if !structure.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &structure.generics,
            "extension structs cannot be generic",
        ));
    }
    let identifier = &structure.ident;
    let default_id = unraw(&identifier.to_string())
        .replace('_', "-")
        .to_ascii_lowercase();
    let id = arguments.id.unwrap_or(default_id);
    let name = arguments
        .name
        .unwrap_or_else(|| title_case(&unraw(&identifier.to_string())));
    let version = option_string(arguments.version);

    Ok(quote! {
        #structure

        #[::ri::__private::async_trait]
        impl ::ri::ext::Extension for #identifier {
            fn descriptor(&self) -> ::ri::ext::ExtensionDescriptor {
                ::ri::ext::ExtensionDescriptor {
                    id: #id.to_owned(),
                    name: #name.to_owned(),
                    version: #version,
                    source: ::ri::ext::SourceInfo::inline(#id),
                }
            }

            async fn register(
                &self,
                __ri_registrar: &mut ::ri::ext::ExtensionRegistrar,
            ) -> ::std::result::Result<(), ::ri::ext::ExtensionInitError> {
                #identifier::register(self, __ri_registrar).await
            }
        }

        impl #identifier {
            #[doc = "Type-erases this value as a native Ri extension."]
            pub fn into_extension(self) -> ::std::sync::Arc<dyn ::ri::ext::Extension>
            where
                Self: 'static,
            {
                ::std::sync::Arc::new(self)
            }
        }
    })
}

fn require_async(function: &ItemFn, kind: &str) -> syn::Result<()> {
    if function.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            function.sig.fn_token,
            format!("{kind} functions must be async"),
        ));
    }
    Ok(())
}

fn require_plain_function(function: &ItemFn) -> syn::Result<()> {
    if !function.sig.generics.params.is_empty()
        || function.sig.constness.is_some()
        || function.sig.unsafety.is_some()
        || function.sig.abi.is_some()
        || function.sig.variadic.is_some()
    {
        return Err(syn::Error::new_spanned(
            &function.sig,
            "generated Ri wrappers require a non-generic safe Rust function",
        ));
    }
    Ok(())
}

fn argument_type(argument: &FnArg) -> syn::Result<&Type> {
    match argument {
        FnArg::Typed(argument) => Ok(&argument.ty),
        FnArg::Receiver(receiver) => Err(syn::Error::new_spanned(
            receiver,
            "free functions cannot have a self receiver",
        )),
    }
}

fn factory_ident(
    override_name: Option<&str>,
    default: &str,
    span: &syn::Ident,
) -> syn::Result<syn::Ident> {
    syn::parse_str::<syn::Ident>(override_name.unwrap_or(default))
        .map_err(|_| syn::Error::new_spanned(span, "factory must be a valid Rust identifier"))
}

fn option_string(value: Option<String>) -> proc_macro2::TokenStream {
    if let Some(value) = value {
        quote! { ::std::option::Option::Some(#value.to_owned()) }
    } else {
        quote! { ::std::option::Option::None }
    }
}

fn doc_text(attributes: &[Attribute]) -> Option<String> {
    let lines = attributes.iter().filter_map(|attribute| {
        if !attribute.path().is_ident("doc") {
            return None;
        }
        let Meta::NameValue(value) = &attribute.meta else {
            return None;
        };
        let Expr::Lit(ExprLit {
            lit: Lit::Str(value),
            ..
        }) = &value.value
        else {
            return None;
        };
        Some(value.value().trim().to_owned())
    });
    let result = lines
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    (!result.is_empty()).then_some(result)
}

fn title_case(value: &str) -> String {
    value
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut characters = part.chars();
            characters.next().map_or_else(String::new, |first| {
                first.to_uppercase().collect::<String>() + characters.as_str()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn pascal_case(value: &str) -> String {
    title_case(value).replace(' ', "")
}

fn unraw(value: &str) -> String {
    value.strip_prefix("r#").unwrap_or(value).to_owned()
}
