import os
os.system('') #to see colors
from dotenv import load_dotenv
load_dotenv()
import json
import weaviate
import asyncio
import argparse

#colors for better logs in terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


async def main():
    parser = argparse.ArgumentParser(description="Evaluator for RAG.")
    parser.add_argument("--path",
                type=str,
                help="Path to question file.")
    
    args = parser.parse_args()
    path = args.path
    
    #load data
    ids = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                if "source_chunk_id" in entry:
                    ids.append(entry["source_chunk_id"])
    
    except FileNotFoundError:
        raise FileNotFoundError(f"\nERROR: File not found: {path}")
    except Exception as e:
        raise Exception(f"\nERROR: while loading file: {path}: {e}")
        

    #connect to db
    client =  weaviate.use_async_with_custom(
            http_host=os.getenv("WEAVIATE_HOST"), http_port=int(os.getenv("WEAVIATE_REST_PORT")), http_secure=False,
            grpc_host=os.getenv("WEAVIATE_HOST"), grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT")), grpc_secure=False,
        )
    
    await client.connect()
    #load data
    chunks = client.collections.get("Chunks")

    #check chunks in db
    counter = 0
    for id in ids:
        try:
            chunk = await chunks.query.fetch_object_by_id(id)
            if chunk:
                counter = counter +1
            else:
                print(f"{Colors.RED} Chunk NOT FOUND, id: {id} {Colors.RESET}")


        except Exception as e:
            raise Exception(f"ERROR: while looking for chunk id: {id}: {e}")

    await client.close()

    if (counter == len(ids)):
        print(f"{Colors.GREEN} ------------- All chunks were FOUND ------------- {Colors.RESET} ")
    

if __name__ == "__main__":
    asyncio.run(main())