from steps.ingest_data import ingest_data

from pipelines.data_fetching import data_fetching

if __name__ == "__main__":
    fetch = data_fetching()
    print("Finished fetching")
