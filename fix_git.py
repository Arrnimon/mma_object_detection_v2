import os
import subprocess

def run_cmd(cmd):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), -1

def main():
    print("=== Fixing Git Push Issue ===")
    
    # Step 1: Check current status
    print("\n1. Checking git status...")
    stdout, stderr, code = run_cmd("git status")
    print(stdout)
    
    # Step 2: Find all MP4 files in the repository
    print("\n2. Finding MP4 files...")
    stdout, stderr, code = run_cmd("git ls-files | findstr /i mp4")
    if stdout:
        print("Found MP4 files:")
        mp4_files = stdout.split('\n')
        for file in mp4_files:
            if file.strip():
                print(f"  - {file}")
        
        # Step 3: Remove MP4 files from git tracking
        print("\n3. Removing MP4 files from git tracking...")
        for file in mp4_files:
            if file.strip():
                print(f"Removing {file} from tracking...")
                run_cmd(f'git rm --cached "{file}"')
    else:
        print("No MP4 files found in git tracking.")
    
    # Step 4: Add all changes
    print("\n4. Adding changes...")
    run_cmd("git add .")
    
    # Step 5: Commit changes
    print("\n5. Committing changes...")
    run_cmd('git commit -m "Remove MP4 files from tracking"')
    
    # Step 6: Try to push
    print("\n6. Attempting to push...")
    stdout, stderr, code = run_cmd("git push")
    if code == 0:
        print("✓ Push successful!")
    else:
        print("✗ Push failed:")
        print(stderr)
        print("\nYou may need to force push or resolve conflicts.")

if __name__ == "__main__":
    main()
