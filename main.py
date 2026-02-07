from dotenv import load_dotenv
import runpy

def main():
    print("Hello from chat-cli!")
    load_dotenv()
    runpy.run_path("Background_Streaming.py", run_name="__main__")


if __name__ == "__main__":
    main()
