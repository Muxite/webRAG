from app.reader import Reader
import asyncio

if __name__ == "__main__":
    reader = Reader()
    asyncio.run(reader.run_worker())