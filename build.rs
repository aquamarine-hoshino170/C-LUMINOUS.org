use std::{env, fs, process::Command};

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // ARMv7 NEON Q8_0 GEMV kernel
    let q8_src = format!("{}/examples/q8_0_neon.c", manifest_dir);

    let q8_obj = format!("{}/q8_0_neon.o", out_dir);

    let q8_lib = format!("{}/libluminous_q8_0_neon.a", out_dir);

    let status = std::process::Command::new("clang")
        .args([
            "-O3",
            "-mfpu=neon",
            "-mfloat-abi=softfp",
            "-c",
            &q8_src,
            "-o",
            &q8_obj,
        ])
        .status()
        .expect("failed to invoke clang for Q8_0");

    assert!(status.success(), "clang failed for q8_0_neon.c");

    let status = std::process::Command::new("llvm-ar")
        .args(["rcs", &q8_lib, &q8_obj])
        .status()
        .expect("failed to invoke llvm-ar for Q8_0");

    assert!(status.success(), "llvm-ar failed for Q8_0");

    println!("cargo:rustc-link-search=native={}", out_dir);

    println!("cargo:rustc-link-lib=static=luminous_q8_0_neon");

    println!("cargo:rerun-if-changed={}", q8_src);

    // ARMv7 NEON Q5_0 GEMV kernel
    let q5_src = format!("{}/examples/q5_0_neon.c", manifest_dir);

    let q5_obj = format!("{}/q5_0_neon.o", out_dir);

    let q5_lib = format!("{}/libluminous_q5_0_neon.a", out_dir);

    let status = std::process::Command::new("clang")
        .args([
            "-O3",
            "-mfpu=neon",
            "-mfloat-abi=softfp",
            "-c",
            &q5_src,
            "-o",
            &q5_obj,
        ])
        .status()
        .expect("failed to invoke clang for Q5_0");

    assert!(status.success(), "clang failed for q5_0_neon.c");

    let status = std::process::Command::new("llvm-ar")
        .args(["rcs", &q5_lib, &q5_obj])
        .status()
        .expect("failed to invoke llvm-ar for Q5_0");

    assert!(status.success(), "llvm-ar failed for Q5_0");

    println!("cargo:rustc-link-search=native={}", out_dir);

    println!("cargo:rustc-link-lib=static=luminous_q5_0_neon");

    println!("cargo:rerun-if-changed={}", q5_src);

    // ARMv7 NEON RMSNorm kernel
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let out_dir = std::env::var("OUT_DIR").unwrap();

    let rms_src = format!("{}/examples/rmsnorm_neon.c", manifest_dir);

    let rms_obj = format!("{}/rmsnorm_neon.o", out_dir);

    let rms_lib = format!("{}/librmsnorm_neon.a", out_dir);

    let status = std::process::Command::new("clang")
        .args([
            "-O3",
            "-mfpu=neon",
            "-mfloat-abi=softfp",
            "-c",
            &rms_src,
            "-o",
            &rms_obj,
        ])
        .status()
        .expect("failed to invoke clang");

    assert!(status.success(), "clang failed for rmsnorm_neon.c");

    let status = std::process::Command::new("llvm-ar")
        .args(["rcs", &rms_lib, &rms_obj])
        .status()
        .expect("failed to invoke llvm-ar");

    assert!(status.success(), "llvm-ar failed for rmsnorm_neon.o");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=rmsnorm_neon");
    println!("cargo:rerun-if-changed={}", rms_src);

    let out = env::var("OUT_DIR").unwrap();
    let src = "examples/neon_dot.c";
    let obj = format!("{}/neon_dot.o", out);
    let lib = format!("{}/libluminous_neon.a", out);

    let status = Command::new("clang")
        .args([
            "-O3",
            "-mfpu=neon",
            "-mfloat-abi=softfp",
            "-c",
            src,
            "-o",
            &obj,
        ])
        .status()
        .expect("failed to execute clang");

    assert!(status.success(), "clang failed");

    let status = Command::new("ar")
        .args(["rcs", &lib, &obj])
        .status()
        .expect("failed to execute ar");

    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out);
    println!("cargo:rustc-link-lib=static=luminous_neon");
    println!("cargo:rerun-if-changed={}", src);

    let _ = fs::metadata(src).expect("neon_dot.c missing");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR missing");

    let q4k_src = std::path::PathBuf::from(&manifest_dir)
        .join("examples")
        .join("q4k_onthefly.c");

    let out_dir = std::path::PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR missing"));

    let q4k_obj = out_dir.join("q4k_onthefly.o");

    let q4k_lib = out_dir.join("libluminous_q4k_neon.a");

    // ============================================================
    // LUMINOUS Q4_K ARMv7 NEON ON-THE-FLY KERNEL
    // ============================================================

    let neon_status = Command::new("clang")
        .args([
            "-O3",
            "-mfpu=neon",
            "-mfloat-abi=softfp",
            "-c",
            q4k_src.to_str().expect("invalid Q4_K source path"),
            "-o",
            q4k_obj.to_str().expect("invalid Q4_K object path"),
        ])
        .status()
        .expect("failed to execute clang for Q4_K NEON");

    assert!(neon_status.success(), "Q4_K NEON clang compilation failed");

    let ar_status = Command::new("ar")
        .args([
            "rcs",
            q4k_lib.to_str().expect("invalid Q4_K library path"),
            q4k_obj.to_str().expect("invalid Q4_K object path"),
        ])
        .status()
        .expect("failed to execute ar for Q4_K NEON");

    assert!(ar_status.success(), "Q4_K NEON archive creation failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());

    println!("cargo:rustc-link-lib=static=luminous_q4k_neon");

    println!("cargo:rerun-if-changed={}", q4k_src.display());
}
