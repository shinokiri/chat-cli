from stream_with_retry import stream_with_retry
from openai import OpenAI, NOT_GIVEN
from prompt_toolkit import PromptSession
from rich import print_json
import argparse

WEB_SEARCH_TOOL = {"type": "web_search"}
PYTHON_TOOL = {
    "type": "code_interpreter",
    "container": {"type": "auto"}
}

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "-t", "--tools",
        action="store_true",
        help="Enable all tools (web_search + code_interpreter)",
    )
    p.add_argument(
        "-s", "--search",
        dest="web_search",
        action="store_true",
        help="Enable web_search tool",
    )
    p.add_argument(
        "-p", "--python",
        dest="python_tool",
        action="store_true",
        help="Enable code_interpreter tool",
    )
    p.add_argument(
        "-i", "--instruction", "--instructions",
        dest="ask_instructions",
        action="store_true",
        help="Ask for developer instructions at start",
    )
    p.add_argument(
        "-c", "--conversation", "--conversations",
        dest="conversation_id",
        help="Reuse an existing conversation by ID",
    )

    return p.parse_args()

def build_tools(args):
    if args.tools or (args.web_search and args.python_tool):
        return [WEB_SEARCH_TOOL, PYTHON_TOOL]
    if args.web_search:
        return [WEB_SEARCH_TOOL]
    if args.python_tool:
        return [PYTHON_TOOL]
    return NOT_GIVEN

def prompt_instructions(session: PromptSession):
    try:
        instructions = session.prompt(message="\nDeveloper:\n").strip() or NOT_GIVEN
        print()
        return instructions
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(0)

def main():
    args = parse_args()
    tools = build_tools(args)
    
    session = PromptSession(
        multiline=True,
        enable_open_in_editor=True,
    )

    instructions = prompt_instructions(session) if args.ask_instructions else NOT_GIVEN

    client = OpenAI()
    conversation = client.conversations.retrieve(args.conversation_id) if args.conversation_id else client.conversations.create()
    print(conversation.id)
    
    while True:
        response_id = None
        sequence_number = 0

        def on_cancel():
            if response_id is not None:
                try:
                    client.responses.cancel(response_id=response_id)
                except Exception:
                    pass

        try:
            input_text = session.prompt(message="\nUser:\n").strip()
            if not input_text:
                continue

            def make_stream():
                if response_id is None:
                    return client.responses.stream(
                        model="gpt-5.2",
                        tools=tools,
                        input=input_text,
                        instructions=instructions,
                        conversation={"id": conversation.id},
                        reasoning={"effort": "xhigh", "summary": "detailed"},
                        text={"verbosity": "high"},
                        service_tier="priority",
                        background=True,
                    )
                else:
                    return client.responses.stream(
                        response_id=response_id,
                        starting_after=sequence_number,
                    )

            for event in stream_with_retry(make_stream):
                sequence_number = event.sequence_number
    
                match event.type:
                    case "response.output_text.delta" | "response.reasoning_summary_text.delta":
                        print(event.delta, end="", flush=True)
    
                    case "response.reasoning_summary_part.done" | "response.content_part.done":
                        print(end="\n\n")
    
                    case "response.created":
                        response_id = event.response.id
                        print(f"\n{response_id}\n\n{event.response.model} is thinking...", end="\n\n")
    
                    case "response.completed":
                        elapsed = event.response.completed_at - event.response.created_at
                        print(f"Elapsed {elapsed:.1f}s.\n")
    
                    case "error":
                        print_json(data=event.model_dump(mode="json"))
                    
                    case _:
                        pass
                    
        except KeyboardInterrupt:
            on_cancel()
            print("\nCancelled (Ctrl-C)\n")
            continue
        except EOFError:
            on_cancel()
            print("\nBye (Ctrl-D)\n")
            break

if __name__ == "__main__":
    main()