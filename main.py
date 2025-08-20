import sys
from takeout.run_bert_from_takeout import run_anomaly_detection

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    results = run_anomaly_detection(file_path)
    print("Results:", results)