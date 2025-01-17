

import os
import subprocess

def main():
    while True:
        print("Select an option:")
        print("A: Run BC_Logistic_regression_no_PCA.py (Folder A)")
        print("B: Run McC_Softmax_regression_PCA.py (Folder B)")
        print("C: Exit the script")

        choice = input("Enter your choice (A, B, or C): ").strip().upper()

        if choice == "A":
            script_path = os.path.join(".", "A", "BC_Logistic_regression_no_PCA.py")
            if os.path.exists(script_path):
                print(f"Running {script_path}...")
                try:
                    result = subprocess.run(["python3", script_path], check=True, capture_output=True, text=True)
                    print(result.stdout)  # Print the script output
                except subprocess.CalledProcessError as e:
                    print(f"Error running the script: {e.stderr}")
            else:
                print(f"Script not found: {script_path}")
        elif choice == "B":
            script_path = os.path.join(".", "B", "McC_Softmax_regression_PCA.py")
            if os.path.exists(script_path):
                print(f"Running {script_path}...")
                try:
                    result = subprocess.run(["python3", script_path], check=True, capture_output=True, text=True)
                    print(result.stdout)  # Print the script output
                except subprocess.CalledProcessError as e:
                    print(f"Error running the script: {e.stderr}")
            else:
                print(f"Script not found: {script_path}")
        elif choice == "C":
            print("Exiting the script. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter A, B, or C.")

if __name__ == "__main__":
    main()

