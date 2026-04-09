
import os
import subprocess
import glob
import sys
import difflib

def is_similar(line1, line2, threshold=0.75):
    """
    Calculates similarity ratio between two strings using SequenceMatcher.
    Returns True if similarity > threshold.
    """
    if not line1 or not line2:
        return False
    # Quick length check optimization: if lengths differ drastically, they aren't similar
    len1, len2 = len(line1), len(line2)
    if len1 == 0 or len2 == 0: return False
    if abs(len1 - len2) / max(len1, len2) > (1 - threshold):
        return False
        
    return difflib.SequenceMatcher(None, line1, line2).ratio() > threshold

def smart_print(lines, similarity_threshold=0.75):
    """
    Prints lines but collapses consecutive similar lines if the streak is >= 5.
    """
    if not lines:
        return

    buffer = []
    prototype = None

    def flush_buffer(buf):
        if not buf:
            return
        
        if len(buf) < 5:
            # Not enough similarity to justify collapsing
            for l in buf:
                print(l)
        else:
            # Collapse!
            # Print first line
            print(buf[0])
            # Print skip message
            count = len(buf) - 2
            if count > 0:
                print(f"   ... [Skipped {count} similar lines] ...")
            # Print last line (if different from first, which it usually is in logs)
            if len(buf) > 1:
                print(buf[-1])

    for line in lines:
        if prototype is None:
            buffer.append(line)
            prototype = line
            continue

        if is_similar(prototype, line, similarity_threshold):
            buffer.append(line)
            # Update prototype to track drifting similarity (e.g. Epoch 1 -> Epoch 2)
            prototype = line 
        else:
            # Similarity broken. Flush current buffer.
            flush_buffer(buffer)
            # Start new buffer with current line
            buffer = [line]
            prototype = line

    # Flush remaining
    flush_buffer(buffer)

def get_line_count(filepath):
    """Returns the number of lines in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def run_script(path, line_count=None):
    print(f"\n{'='*60}")
    header = f"RUNNING: {os.path.basename(path)}"
    if line_count is not None:
        header += f" ({line_count} lines)"
    print(header)
    print(f"{'='*60}")
    
    try:
        # Force UTF-8 for subprocess to handle emojis (█, 🚨)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["ODYSSNET_DISABLE_PLOT"] = "1"  # Disable interactive plotting in subprocesses
        
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            encoding='utf-8',       # FORCE READ AS UTF-8
            errors='replace',       # Handle non-utf8 characters gracefully
            env=env
        )
        
        output_lines = result.stdout.splitlines()
        
        # Use Smart Print Logic
        smart_print(output_lines, similarity_threshold=0.75)
                
        if result.returncode != 0:
            print(f"\n❌ FAILED with code {result.returncode}")
            print("ERROR OUTPUT:")
            print(result.stderr)
        else:
            print("\n✅ DONE")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    poc_scripts = glob.glob(os.path.join(base_dir, "convergence_*.py"))
    exp_scripts = glob.glob(os.path.join(base_dir, "advanced", "convergence_*.py"))
    
    all_scripts = poc_scripts + exp_scripts
    
    # Filter out Nothing. Run EVERYTHING.
    blacklist = []
    
    targets = []
    for s in all_scripts:
        filename = os.path.basename(s)
        name = os.path.splitext(filename)[0]
            
        if name not in blacklist:
            targets.append(s)
            
    # Calculate line counts and sort
    targets_with_counts = []
    for s in targets:
        count = get_line_count(s)
        targets_with_counts.append((count, s))
    
    # Sort by line count (lowest to highest)
    targets_with_counts.sort(key=lambda x: x[0])
    
    print(f"Found {len(targets_with_counts)} scripts to run:")
    for count, script in targets_with_counts:
        print(f" - {os.path.basename(script):<40} | {count:4d} lines")
    
    for count, script in targets_with_counts:
        run_script(script, line_count=count)

if __name__ == "__main__":
    main()
