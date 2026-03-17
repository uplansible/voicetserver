use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/kernels/m1_attention.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");

    let mut builder = KernelBuilder::new()
        .source_dir("src/kernels")
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // MSVC needs _USE_MATH_DEFINES for math constants
    if let Ok(target) = env::var("TARGET") {
        if target.contains("msvc") {
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    let bindings = builder.build_ptx()?;
    bindings.write(&ptx_path)?;

    Ok(())
}
