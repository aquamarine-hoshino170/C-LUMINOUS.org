import struct
import numpy as np
import gguf

MODEL_PATH = "model.gguf"  # এখানে আপনার gguf ফাইলের নাম/পাথ দিন

print("[*] GGUF ফাইল লোড করা হচ্ছে...")
try:
    reader = gguf.GGUFReader(MODEL_PATH)
except Exception as e:
    print(f"[-] ফাইল লোড করতে সমস্যা: {e}")
    exit(1)

tensor_name = "token_embd.weight"
tensor = next((t for t in reader.tensors if t.name == tensor_name), None)

if tensor is None:
    print(f"[-] {tensor_name} পাওয়া যায়নি!")
    exit(1)

print(f"[+] টেনসর পাওয়া গেছে: {tensor.name}")
print(f"    টাইপ: {tensor.tensor_type} (GGML Type 6 = Q5_0)")
print(f"    শেপ: {tensor.shape}")

raw_bytes = bytes(tensor.data)
print(f"[+] মোট বাইট সাইজ: {len(raw_bytes)}")

# Q5_0 স্পেসিফিকেশন: প্রতিটি ব্লকে ৩২টি ফ্লোট থাকে
# সাইজ: ২ বাইট (fp16 scale 'd') + ৪ বাইট (qh - উচ্চ বিট) + ১৬ বাইট (qs - নিম্ন ৪ বিট) = ২২ বাইট
BLOCK_SIZE = 22

def decode_q5_0_block(block_bytes):
    # স্কেল d (float16) এক্সট্র্যাক্ট করা
    d = np.frombuffer(block_bytes[:2], dtype=np.float16).astype(np.float32)[0]
    
    # qh (uint32) - ৩২টি মানের জন্য অতিরিক্ত ৫ম বিট
    qh = struct.unpack("<I", block_bytes[2:6])[0]
    
    # qs - ১৬ বাইট (প্রতি বাইটে দুটি ৪-বিট মান)
    qs = np.frombuffer(block_bytes[6:22], dtype=np.uint8)
    
    values = np.zeros(32, dtype=np.float32)
    for i in range(16):
        b = qs[i]
        # নিম্ন ৪ বিট ও উচ্চ ৪ বিট আলাদা করা
        q0 = b & 0x0F
        q1 = b >> 4
        
        # qh থেকে ৫ম বিট যোগ করা
        h0 = ((qh >> i) & 1) << 4
        h1 = ((qh >> (i + 16)) & 1) << 4
        
        # মানগুলোকে সেন্টারিং করা (-16 বিয়োগ করে)
        val0 = (q0 | h0) - 16
        val1 = (q1 | h1) - 16
        
        values[i] = val0 * d
        values[i + 16] = val1 * d
        
    return d, values

print("\n[*] প্রথম ব্লকের বাইট ডিকোড টেস্ট...")
first_block = raw_bytes[:BLOCK_SIZE]
d_scale, decoded_32 = decode_q5_0_block(first_block)

print(f"    স্কেল ভ্যালু (d): {d_scale}")
print("    প্রথম ৩২টি ডিকোড করা মান:")
for i, v in enumerate(decoded_32):
    print(f"    x[{i:02d}] = {v:.6f}")

rms = np.sqrt(np.mean(decoded_32 ** 2))
print(f"\n[+] ম্যানুয়াল ক্যালকুলেটেড RMS: {rms:.6f}")
print(f"[+] Min: {decoded_32.min():.6f}, Max: {decoded_32.max():.6f}")

if abs(rms) > 50.0:
    print("\n[!] সতর্কতা: RMS অস্বাভাবিক বেশি (> ৫০)! ফাইলটি করাপ্ট অথবা অফসেট ভুল।")
else:
    print("\n[✓] টেস্ট সফল! মানগুলো স্বাভাবিক রেঞ্জে আছে।")
