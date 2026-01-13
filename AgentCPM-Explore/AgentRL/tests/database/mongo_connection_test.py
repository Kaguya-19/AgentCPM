import beanie
import asyncio

c_str = "mongodb://root:password@g8:27017/test?authSource=admin"

class TestDocument(beanie.Document):
    name: str

    class Settings:
        collection = "test_collection"

async def main():
    await beanie.init_beanie(
        connection_string=c_str,
        document_models=[TestDocument],
    )
    await TestDocument(name="test").save()
    

if __name__ == "__main__":
    asyncio.run(main())
    