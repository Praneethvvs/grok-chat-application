import subprocess

if __name__ == "__main__":
    cmd = ["uvicorn", "azure_function:app", "--host", "0.0.0.0", "--port", "8080"]
    subprocess.run(cmd)
