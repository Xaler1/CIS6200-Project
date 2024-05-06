from openai import OpenAI
from keys import openai_key
import time
import json

client = OpenAI(api_key=openai_key)
for i in range(0, 20):
    print("Processing batch", i, "at", time.strftime("%H:%M:%S"))
    response = client.files.create(
        file=open(f"train_job{i}.jsonl", "rb"),
        purpose="batch"
    )
    file_id = response.id
    response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    batch_id = response.id
    batch_complete = False
    print("Created batch", batch_id, "at", time.strftime("%H:%M:%S"))
    while not batch_complete:
        time.sleep(60 * 5)
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            batch_complete = True
            print("Batch", batch_id, "completed at", time.strftime("%H:%M:%S"))
            output_file_id = batch.output_file_id
            data = client.files.content(output_file_id)
            with open(f"result{i}.jsonl", "wb") as f:
                f.write(data.content)
            with open(f"result{i}.jsonl", "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = json.loads(line)
                    id = line["custom_id"].split("-")[1]
                    annotation = line["response"]["body"]["choices"][0]["message"]["content"]
                    with open(f"train_split/{id}.txt", "w") as f:
                        f.write(annotation)

        elif batch.status == "failed":
            batch_complete = True
            print("Batch", batch_id, "failed at", time.strftime("%H:%M:%S"))