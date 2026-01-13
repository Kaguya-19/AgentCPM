import asyncio
from fastmcp.client import Client

task = {"task_config": {"id": "550ce7e7-747b-495f-b122-acdc4d0b8e54", "snapshot": "libreoffice_impress", "instruction": "I am checking our soccer club's to-do list for the last semester and adding strike-through sign on the line we have already accomplished. Could you help me add a strike-through on the first and second line?", "source": "https://technical-tips.com/blog/software/text-in-libreoffice-strikethrough--6948#:~:text=To%20strikethrough%20Text%20in%20LibreOffice%201%20In%20your,effect%22%20can%20your%20additionally%2C%20for%20example%2C%20double%20underline.", "config": [{"type": "download", "parameters": {"files": [{"url": "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/550ce7e7-747b-495f-b122-acdc4d0b8e54/New_Club_Spring_2018_Training.data", "path": "/home/user/Desktop/New_Club_Spring_2018_Training.pptx"}]}}, {"type": "open", "parameters": {"path": "/home/user/Desktop/New_Club_Spring_2018_Training.pptx"}}, {"type": "sleep", "parameters": {"seconds": 0.5}}, {"type": "execute", "parameters": {"command": ["python", "-c", "import pyautogui; import time;  time.sleep(4); pyautogui.click(170, 250); time.sleep(1);pyautogui.press('down'); time.sleep(1); pyautogui.press('down'); time.sleep(1); pyautogui.press('down'); time.sleep(1); pyautogui.press('down'); time.sleep(1); "]}}], "trajectory": "trajectories/", "related_apps": ["libreoffice_impress"], "evaluator": {"postconfig": [{"type": "activate_window", "parameters": {"window_name": "New_Club_Spring_2018_Training.pptx - LibreOffice Impress", "strict": True}}, {"type": "sleep", "parameters": {"seconds": 0.5}}, {"type": "execute", "parameters": {"command": ["python", "-c", "import pyautogui; import time; pyautogui.hotkey('ctrl', 's'); time.sleep(0.5);"]}}, {"type": "sleep", "parameters": {"seconds": 0.5}}], "func": ["compare_pptx_files", "compare_pptx_files"], "conj": "or", "expected": [{"type": "cloud_file", "path": "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/550ce7e7-747b-495f-b122-acdc4d0b8e54/New_Club_Spring_2018_Training_with_strike.data", "dest": "New_Club_Spring_2018_Training_Gold.pptx"}, {"type": "cloud_file", "path": "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/550ce7e7-747b-495f-b122-acdc4d0b8e54/New_Club_Spring_2018_Training_Gold_all_fonts_1.pptx", "dest": "New_Club_Spring_2018_Training_Gold_all_fonts_1.pptx"}], "result": [{"type": "vm_file", "path": "/home/user/Desktop/New_Club_Spring_2018_Training.pptx", "dest": "New_Club_Spring_2018_Training.pptx"}, {"type": "vm_file", "path": "/home/user/Desktop/New_Club_Spring_2018_Training.pptx", "dest": "New_Club_Spring_2018_Training.pptx"}]}, "proxy": True, "fixed_ip": False, "possibility_of_env_change": "low"}, "instruction": "I am checking our soccer club's to-do list for the last semester and adding strike-through sign on the line we have already accomplished. Could you help me add a strike-through on the first and second line?"}

# python server.py  --db-connection-string  mongodb://admin:2025AgentRL@172.16.1.37:27021/osworld?authSource=admin --oss-connection-string  minio://eLaKRaUieEsndqq4:CVPYjmPXdjV4Z40iEMcBwviviMcZHjQ0@172.16.1.37:11210/trajectory?secure=false --host 127.0.0.1 --port 10238

async def main():
    import time
    s_time = time.time()
    async with Client(
        "http://127.0.0.1:47235/mcp"
    ) as client:
    
        # res = await client.call_tool(
        #     "reset_env",
        #     {
        #         "task_config":task["task_config"]
        #     }
        # )
        # print(res)
        
        res = await client.ping()
        print("Ping:", res)

        res = await client.read_resource(
            "resource://screenshot/url",
        )
        print(res)

        # res = await client.read_resource(
        #     "resource://evaluate"
        # )
        # print(res)

        # res = await client.call_tool(
        #     "terminate_env",
        #     {}
        # )
        # print(res)
    print(f"Total time: {time.time() - s_time}")
async def multi_main():
    tasks = []
    # for _ in range(4):
    await main()
    #     tasks.append(main())
    # await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())