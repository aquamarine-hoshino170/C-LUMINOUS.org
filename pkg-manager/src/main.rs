use std::env;
use std::fs;
use std::process::Command;

const DEFAULT_ENTRY: &str = "src/main.lum";
const MANIFEST_FILE: &str = "luminous.json";

fn init_project(proj_name: &str) {
    println!("\x1b[1;36mInitializing Luminous project: \x1b[1;32m{}\x1b[0m...", proj_name);
    
    // ডিরেক্টরি তৈরি
    if let Err(e) = fs::create_dir_all("src") {
        eprintln!("\x1b[1;31mError creating src/ directory: {}\x1b[0m", e);
        return;
    }

    // luminous.json মেনিফেস্ট তৈরি
    let manifest_content = format!(
r#"{{
  "name": "{}",
  "version": "0.1.0",
  "description": "A Luminous project",
  "main": "{}",
  "dependencies": {{}}
}}"#,
        proj_name, DEFAULT_ENTRY
    );

    if let Err(e) = fs::write(MANIFEST_FILE, manifest_content) {
        eprintln!("\x1b[1;31mError creating {}: {}\x1b[0m", MANIFEST_FILE, e);
        return;
    }

    // src/main.lum স্টার্টার ফাইল তৈরি
    let starter_code = format!(
r#"# {} - Entry Point
print("========================================")
print("  🚀 Welcome to {} (Luminous Project)")
print("========================================")

today version is "0.1.0"
print("Project initialized successfully on version: ${{version}}")
"#,
        proj_name, proj_name
    );

    if let Err(e) = fs::write(DEFAULT_ENTRY, starter_code) {
        eprintln!("\x1b[1;31mError creating {}: {}\x1b[0m", DEFAULT_ENTRY, e);
        return;
    }

    println!("\x1b[1;32m✓ Created {}\x1b[0m", MANIFEST_FILE);
    println!("\x1b[1;32m✓ Created {}\x1b[0m", DEFAULT_ENTRY);
    println!("\x1b[1;35m🎉 Project '{}' ready! Run with: luminous-pkg run\x1b[0m", proj_name);
}

fn run_project() {
    if !std::path::Path::new(MANIFEST_FILE).exists() {
        eprintln!("\x1b[1;31mError: No '{}' found in current directory.\x1b[0m", MANIFEST_FILE);
        eprintln!("Run '\x1b[1;33mluminous-pkg init\x1b[0m' first.");
        return;
    }

    let manifest_data = match fs::read_to_string(MANIFEST_FILE) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\x1b[1;31mFailed to read {}: {}\x1b[0m", MANIFEST_FILE, e);
            return;
        }
    };

    // বেসিক স্ট্রিং পার্সিং দিয়ে এন্ট্রি ফাইল বের করা
    let mut main_file = DEFAULT_ENTRY.to_string();
    for line in manifest_data.lines() {
        if line.contains("\"main\":") {
            if let Some(start) = line.find(':') {
                let raw_val = line[start + 1..].replace(['"', ',', ' '], "");
                if !raw_val.is_empty() {
                    main_file = raw_val;
                }
            }
        }
    }

    if !std::path::Path::new(&main_file).exists() {
        eprintln!("\x1b[1;31mError: Entry file '{}' not found!\x1b[0m", main_file);
        return;
    }

    println!("\x1b[1;34m⚡ Executing [{}]...\x1b[0m\n", main_file);
    
    let status = Command::new("luminous")
        .arg(&main_file)
        .status();

    match status {
        Ok(st) => {
            if !st.success() {
                eprintln!("\n\x1b[1;31mProcess exited with status code: {}\x1b[0m", st);
            }
        }
        Err(e) => {
            eprintln!("\x1b[1;31mFailed to launch luminous runtime: {}\x1b[0m", e);
            eprintln!("Ensure 'luminous' binary is in your PATH.");
        }
    }
}

fn print_help() {
    println!("\x1b[1;36m🌊 luminous-pkg: The Official Luminous Package Manager 🌊\x1b[0m");
    println!("Usage: luminous-pkg <command> [arguments]\n");
    println!("Commands:");
    println!("  \x1b[1;32minit [name]\x1b[0m     Initialize a new Luminous project in the current directory");
    println!("  \x1b[1;32mrun\x1b[0m             Run the current project defined in luminous.json");
    println!("  \x1b[1;32mhelp\x1b[0m            Show this help manual");
    println!("  \x1b[1;32mversion\x1b[0m         Show luminous-pkg version");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_help();
        return;
    }

    match args[1].as_str() {
        "init" => {
            let name = if args.len() >= 3 {
                &args[2]
            } else {
                "my_luminous_app"
            };
            init_project(name);
        }
        "run" => {
            run_project();
        }
        "version" | "-v" | "--version" => {
            println!("luminous-pkg v1.0.0");
        }
        "help" | "-h" | "--help" => {
            print_help();
        }
        unknown => {
            eprintln!("\x1b[1;31mUnknown command: '{}'\x1b[0m", unknown);
            println!("Run 'luminous-pkg help' for usage instructions.");
        }
    }
}
