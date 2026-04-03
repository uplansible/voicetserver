fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/kernels/m1_attention.cu");

    // CUDA kernel compilation only runs when the `cuda` feature is enabled.
    // Dev builds (Docker, no GPU) skip this block entirely.
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        compile_cuda_kernels().expect("CUDA kernel compilation failed");
    }
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
