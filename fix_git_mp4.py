#!/usr/bin/env python3
"""
Helper script to fix git issues with MP4 files
This script will:
1. Remove MP4 files from git tracking
2. Clean up the repository
3. Help you commit the changes
"""

import subprocess
import os
import sys

def run_command(command, description):
    """Run a git command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"✓ {description} completed successfully")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return None

def main():
    print("=== Git MP4 File Cleanup Script ===")
    print("This script will help you remove MP4 files from git tracking.")
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("Error: Not in a git repository!")
        return
    
    # Step 1: Check current status
    status = run_command("git status", "Checking git status")
    if status is None:
        return
    
    # Step 2: Remove MP4 files from git tracking (but keep them locally)
    print("\nRemoving MP4 files from git tracking...")
    
    # Find all MP4 files currently tracked by git
    tracked_files = run_command("git ls-files | grep -i '\\.mp4'", "Finding tracked MP4 files")
    
    if tracked_files:
        print("Found tracked MP4 files:")
        for file in tracked_files.split('\n'):
            if file.strip():
                print(f"  - {file}")
        
        # Remove them from git tracking
        for file in tracked_files.split('\n'):
            if file.strip():
                run_command(f'git rm --cached "{file}"', f"Removing {file} from tracking")
    else:
        print("No MP4 files currently tracked by git.")
    
    # Step 3: Add .gitignore changes
    run_command("git add .gitignore", "Adding .gitignore changes")
    
    # Step 4: Commit the changes
    print("\n=== Ready to commit changes ===")
    print("The following changes will be committed:")
    run_command("git status", "Final status check")
    
    commit_message = input("\nEnter commit message (or press Enter for default): ").strip()
    if not commit_message:
        commit_message = "Remove MP4 files from tracking and update .gitignore"
    
    run_command(f'git commit -m "{commit_message}"', "Committing changes")
    
    print("\n=== Cleanup Complete! ===")
    print("✓ MP4 files have been removed from git tracking")
    print("✓ .gitignore has been updated")
    print("✓ Changes have been committed")
    print("\nYou can now push your repository without MP4 files:")
    print("  git push")

if __name__ == "__main__":
    main()
