import json


def save_chunks(all_chunks, path="database/chunks.json"):# path is create the json file to store the chunks

    with open(path, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, ensure_ascii=False, indent=4)

    print("Chunks saved successfully!")


def load_chunks(path="chunks.json"):

    with open(path, "r", encoding="utf-8") as file:
        all_chunks = json.load(file)

    print("Chunks loaded successfully!")

    return all_chunks
