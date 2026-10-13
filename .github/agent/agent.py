import os, sys, json, time, pathlib, subprocess, urllib.request, urllib.error

KEY = os.environ["GEMINI_API_KEY"]
MODEL = os.environ.get("MODEL", "gemini-2.5-pro")
TASK = os.environ.get("TASK", "").strip()
MAX_STEPS = int(os.environ.get("MAX_STEPS", "40"))
ROOT = pathlib.Path.cwd().resolve()
BASE = "https://generativelanguage.googleapis.com/v1beta"
SYSTEM = (
    "You are an autonomous senior software engineer working in a git repo (Rust-heavy). "
    "Explore first (list_dir, grep, read_file), plan, then make minimal correct edits with edit_file/write_file. "
    "Verify with run_shell (cargo fmt, cargo clippy -- -D warnings, cargo test). If it fails, read the error and fix it. "
    "Never fake success. When done, call finish with a short markdown summary of what changed and what is unverified."
)

def safe(p):
    q = (ROOT / p).resolve()
    if q != ROOT and ROOT not in q.parents:
        raise ValueError("path outside repo")
    if ".git" in q.relative_to(ROOT).parts[:1]:
        raise ValueError(".git is off limits")
    return q

def list_dir(path="."):
    return sorted(p.name + ("/" if p.is_dir() else "") for p in safe(path).iterdir())[:300]

def read_file(path, start=1, end=400):
    lines = safe(path).read_text(errors="replace").splitlines()
    return "\n".join(f"{i}: {l}" for i, l in enumerate(lines[start-1:end], start))

def write_file(path, content):
    p = safe(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content)
    return f"wrote {len(content)} chars to {path}"

def edit_file(path, old, new):
    p = safe(path); s = p.read_text()
    if s.count(old) != 1:
        return {"error": f"old text must occur exactly once, found {s.count(old)}"}
    p.write_text(s.replace(old, new)); return "ok"

def grep(pattern, path="."):
    r = subprocess.run(["grep", "-rnI", "--exclude-dir=.git", "--exclude-dir=target", pattern, str(safe(path))],
                       capture_output=True, text=True, cwd=ROOT)
    return r.stdout[:6000] or "no matches"

def run_shell(command, timeout=300):
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GH_TOKEN", "GITHUB_TOKEN")}
    try:
        r = subprocess.run(command, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)
        return {"exit": r.returncode, "output": (r.stdout + r.stderr)[-6000:]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

def finish(summary):
    pathlib.Path("/tmp/agent_summary.md").write_text(summary); return "done"

TOOLS = {f.__name__: f for f in (list_dir, read_file, write_file, edit_file, grep, run_shell, finish)}

def decl(name, desc, props, req):
    return {"name": name, "description": desc,
            "parameters": {"type": "OBJECT", "properties": {k: {"type": v} for k, v in props.items()}, "required": req}}

DECLS = [
    decl("list_dir", "List a directory", {"path": "STRING"}, []),
    decl("read_file", "Read file with line numbers", {"path": "STRING", "start": "INTEGER", "end": "INTEGER"}, ["path"]),
    decl("write_file", "Create or overwrite a file", {"path": "STRING", "content": "STRING"}, ["path", "content"]),
    decl("edit_file", "Replace one exact unique snippet in a file", {"path": "STRING", "old": "STRING", "new": "STRING"}, ["path", "old", "new"]),
    decl("grep", "Recursive regex search", {"pattern": "STRING", "path": "STRING"}, ["pattern"]),
    decl("run_shell", "Run a bash command in the repo root", {"command": "STRING", "timeout": "INTEGER"}, ["command"]),
    decl("finish", "Finish the task with a markdown summary", {"summary": "STRING"}, ["summary"]),
]

def call(contents):
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": contents,
            "tools": [{"functionDeclarations": DECLS}], "generationConfig": {"temperature": 0.2}}
    for attempt in range(4):
        req = urllib.request.Request(f"{BASE}/models/{MODEL}:generateContent", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
        try:
            return json.load(urllib.request.urlopen(req, timeout=300))
        except (ConnectionError, urllib.error.URLError, TimeoutError) as e:
            if attempt < 3:
                time.sleep(20 * (attempt + 1)); continue
            print(f"NETWORK ERROR: {e}"); sys.exit(1)
        except urllib.error.HTTPError as e:
            err = e.read().decode()
            if e.code in (429, 500, 503) and attempt < 3:
                time.sleep(20 * (attempt + 1)); continue
            print(f"API ERROR {e.code}: {err}")
            if e.code == 404:
                m = urllib.request.Request(f"{BASE}/models", headers={"x-goog-api-key": KEY})
                for x in json.load(urllib.request.urlopen(m)).get("models", []):
                    if "generateContent" in x.get("supportedGenerationMethods", []):
                        print("AVAILABLE:", x["name"])
            sys.exit(1)

if not TASK:
    sys.exit("no task given")
contents = [{"role": "user", "parts": [{"text": TASK}]}]
for step in range(MAX_STEPS):
    content = call(contents)["candidates"][0].get("content", {})
    contents.append(content)
    parts = content.get("parts", [])
    for p in parts:
        if "text" in p: print("AGENT:", p["text"])
    calls = [p["functionCall"] for p in parts if "functionCall" in p]
    if not calls: break
    results, done = [], False
    for c in calls:
        print(f"> {c['name']} {json.dumps(c.get('args', {}), ensure_ascii=False)[:200]}")
        try: out = TOOLS[c["name"]](**c.get("args", {}))
        except Exception as e: out = {"error": str(e)}
        results.append({"functionResponse": {"name": c["name"], "response": {"result": out}}})
        done = done or c["name"] == "finish"
    contents.append({"role": "user", "parts": results})
    if done: break
else:
    print("stopped: max steps reached")
