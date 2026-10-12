import struct

PATH = "../models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    if n > 10_000_000:
        raise ValueError(f"invalid GGUF string length: {n}")
    return f.read(n).decode("utf-8", errors="replace")

def skip_value(f, typ):
    # GGUF metadata types:
    # 0 UINT8
    # 1 INT8
    # 2 UINT16
    # 3 INT16
    # 4 UINT32
    # 5 INT32
    # 6 FLOAT32
    # 7 BOOL
    # 8 STRING
    # 9 ARRAY
    # 10 UINT64
    # 11 INT64
    # 12 FLOAT64

    sizes = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }

    if typ in sizes:
        f.seek(sizes[typ], 1)
        return

    if typ == 8:
        read_str(f)
        return

    if typ == 9:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]

        if count > 10_000_000:
            raise ValueError(f"invalid GGUF array length: {count}")

        for _ in range(count):
            skip_value(f, elem_type)
        return

    raise ValueError(f"unknown GGUF metadata type: {typ}")

def read_value(f, typ):
    if typ == 0:
        return struct.unpack("<B", f.read(1))[0]
    if typ == 1:
        return struct.unpack("<b", f.read(1))[0]
    if typ == 2:
        return struct.unpack("<H", f.read(2))[0]
    if typ == 3:
        return struct.unpack("<h", f.read(2))[0]
    if typ == 4:
        return struct.unpack("<I", f.read(4))[0]
    if typ == 5:
        return struct.unpack("<i", f.read(4))[0]
    if typ == 6:
        return struct.unpack("<f", f.read(4))[0]
    if typ == 7:
        return bool(struct.unpack("<B", f.read(1))[0])
    if typ == 8:
        return read_str(f)
    if typ == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if typ == 11:
        return struct.unpack("<q", f.read(8))[0]
    if typ == 12:
        return struct.unpack("<d", f.read(8))[0]

    if typ == 9:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        if count > 10_000_000:
            raise ValueError(f"invalid GGUF array length: {count}")
        return [read_value(f, elem_type) for _ in range(count)]

    raise ValueError(f"unknown GGUF metadata type: {typ}")

with open(PATH, "rb") as f:
    magic = f.read(4)

    if magic != b"GGUF":
        raise RuntimeError(f"not a GGUF file: {magic!r}")

    version, tensors, metadata = struct.unpack("<IQQ", f.read(20))

    print("=" * 64)
    print("        LUMINOUS QWEN ARCHITECTURE PROBE")
    print("=" * 64)
    print(f"GGUF version       : {version}")
    print(f"Tensor count       : {tensors}")
    print(f"Metadata count     : {metadata}")
    print()

    found = {}

    for _ in range(metadata):
        key = read_str(f)
        typ = struct.unpack("<I", f.read(4))[0]

        value = read_value(f, typ)

        if (
            key.startswith("general.")
            or key.startswith("qwen2.")
            or "rope" in key.lower()
            or "attention" in key.lower()
        ):
            found[key] = value

    for key in sorted(found):
        print(f"{key:52} = {found[key]}")

    print()
    print("=" * 64)
    print("DERIVED ARCHITECTURE")
    print("=" * 64)

    def get(*keys):
        for k in keys:
            if k in found:
                return found[k]
        return None

    hidden = get(
        "qwen2.embedding_length",
        "qwen2.context_length",
    )

    q_heads = get("qwen2.attention.head_count")
    kv_heads = get("qwen2.attention.head_count_kv")
    layers = get("qwen2.block_count")
    intermediate = get("qwen2.feed_forward_length")
    rope_dim = get("qwen2.rope.dimension_count")
    rope_theta = get("qwen2.rope.freq_base")

    print(f"layers             : {layers}")
    print(f"hidden size        : {hidden}")
    print(f"FFN size           : {intermediate}")
    print(f"Q heads            : {q_heads}")
    print(f"KV heads           : {kv_heads}")

    if hidden and q_heads:
        print(f"head dimension     : {hidden // q_heads}")

    if q_heads and kv_heads:
        print(f"GQA ratio          : {q_heads // kv_heads}")

    print(f"RoPE dimension     : {rope_dim}")
    print(f"RoPE frequency     : {rope_theta}")

    print()
    print("[✓] GGUF metadata parsed successfully")
    print("[✓] Architecture probe complete")
