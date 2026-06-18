fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/kernels/m1_attention.cu");

    // CUDA kernel compilation only runs when the `cuda` feature is enabled.
    // Dev builds (Docker, no GPU) skip this block entirely.
    // VOICETSERVER_BUILD_VARIANT is baked into the `--version` string so the
    // binary can report "CUDA <ver>" or "CPU" without touching the GPU at runtime.
    let variant = if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        compile_cuda_kernels().expect("CUDA kernel compilation failed");
        // cudarc's `dynamic-loading` (vs `dynamic-linking`) no longer emits the CUDA
        // library search path, but candle-kernels / flash-attn still link `-lcudart`.
        // Re-add it so the toolkit runtime resolves at link time. Only the driver
        // (libcuda.so.1) becomes lazily loaded, so `--version` works without a GPU.
        emit_cuda_link_search();
        format!("CUDA {}", detect_cuda_version().unwrap_or_else(|| "unknown".to_string()))
    } else {
        "CPU".to_string()
    };
    println!("cargo::rustc-env=VOICETSERVER_BUILD_VARIANT={variant}");
}

/// Emit `rustc-link-search` for the CUDA runtime libraries so `-lcudart` resolves.
/// Uses CUDA_PATH if set, else the conventional /usr/local/cuda; tries lib64 then lib.
fn emit_cuda_link_search() {
    let base = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    for sub in ["lib64", "lib"] {
        let dir = format!("{base}/{sub}");
        if std::path::Path::new(&dir).is_dir() {
            println!("cargo::rustc-link-search=native={dir}");
        }
    }
}

/// Parse the CUDA toolkit version from `nvcc --version` (e.g. "12.6").
/// Returns None if nvcc is absent or unparseable.
fn detect_cuda_version() -> Option<String> {
    let out = std::process::Command::new("nvcc").arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    // Line looks like: "Cuda compilation tools, release 12.6, V12.6.20"
    let marker = "release ";
    let start = text.find(marker)? + marker.len();
    let rest = &text[start..];
    let end = rest.find(',').unwrap_or(rest.len());
    Some(rest[..end].trim().to_string())
}

fn compile_cuda_kernels() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;
    use std::path::PathBuf;

    // cudaforge is only present when cuda feature is active; import dynamically
    // via the re-exported types. This function is only called with cuda enabled.
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let ptx_path = out_dir.join("ptx.rs");

    // Build PTX via cudaforge (available because cuda feature pulls it in as build-dep)
    let mut builder = cudaforge::KernelBuilder::new()
        .source_dir("src/kernels")
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    if let Ok(target) = env::var("TARGET") {
        if target.contains("msvc") {
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    let bindings = builder.build_ptx()?;
    bindings.write(&ptx_path)?;
    Ok(())
}
