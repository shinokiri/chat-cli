from stream_with_retry import stream_with_retry
from openai import OpenAI, NOT_GIVEN
from prompt_toolkit import PromptSession
from rich import print_json
import argparse
import hashlib
import mimetypes
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

WEB_SEARCH_TOOL = {"type": "web_search"}
PYTHON_TOOL = {
    "type": "code_interpreter",
    "container": {"type": "auto"}
}
PRIMARY_UPLOAD_PURPOSE = "user_data"


@dataclass(frozen=True)
class AttachmentSpec:
    kind: str
    source: str


@dataclass(frozen=True)
class ParsedTurn:
    parts: list[str | AttachmentSpec]
    show_help: bool


def print_slash_help():
    print(
        "\nCommands:\n"
        "  /image <path_or_url>  Attach an image (local file or http/https URL).\n"
        "  /file <path>          Attach a local file.\n"
        "  /help                 Show this help.\n"
        "  //...                 Escape slash command parsing for this line.\n"
    )


def parse_turn_input(raw_input: str) -> ParsedTurn:
    parts: list[str | AttachmentSpec] = []
    text_lines: list[str] = []
    show_help = False

    def flush_text():
        if text_lines:
            parts.append("".join(text_lines))
            text_lines.clear()

    for line in raw_input.splitlines(keepends=True):
        lstripped = line.lstrip()
        if lstripped.startswith("//"):
            leading_ws = line[: len(line) - len(lstripped)]
            text_lines.append(leading_ws + "/" + lstripped[2:])
            continue

        stripped = line.strip()
        if not stripped.startswith("/"):
            text_lines.append(line)
            continue

        try:
            tokens = shlex.split(stripped)
        except ValueError:
            # Keep unparseable slash-lines as plain text.
            text_lines.append(line)
            continue

        if not tokens:
            text_lines.append(line)
            continue

        command = tokens[0].lower()
        if command == "/help":
            show_help = True
            continue

        if command in {"/image", "/file"} and len(tokens) >= 2:
            source = " ".join(tokens[1:]).strip()
            if source:
                flush_text()
                parts.append(
                    AttachmentSpec(
                        kind="image" if command == "/image" else "file",
                        source=source,
                    )
                )
                continue

        # Unknown slash command lines are treated as normal text.
        text_lines.append(line)

    flush_text()

    return ParsedTurn(
        parts=parts,
        show_help=show_help,
    )


def is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def resolve_local_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


def ensure_readable_file(path: Path):
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"File is not readable: {path}")


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def compute_sha256_and_size(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    total_bytes = 0
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
            total_bytes += len(chunk)
    return hasher.hexdigest(), total_bytes


def compute_stream_sha256_and_size(binary_content) -> tuple[str, int]:
    hasher = hashlib.sha256()
    total_bytes = 0
    for chunk in binary_content.iter_bytes():
        if not chunk:
            continue
        hasher.update(chunk)
        total_bytes += len(chunk)
    return hasher.hexdigest(), total_bytes


def is_content_download_forbidden(error: Exception) -> bool:
    message = str(error)
    return "Not allowed to download files of purpose" in message


def create_uploaded_file(
    client: OpenAI,
    path: Path,
    *,
    filename: str,
    mime_type: str,
    purpose: str,
):
    with path.open("rb") as file_obj:
        return client.files.create(
            file=(filename, file_obj, mime_type),
            purpose=purpose,
        )


def verify_uploaded_content(
    client: OpenAI,
    *,
    file_id: str,
    filename: str,
    expected_sha256: str,
    expected_size: int,
):
    remote_sha256, remote_size = compute_stream_sha256_and_size(client.files.content(file_id))
    if remote_sha256 != expected_sha256 or remote_size != expected_size:
        raise ValueError(
            f"Integrity check failed for {filename} "
            f"(local sha256={expected_sha256}, remote sha256={remote_sha256}, "
            f"local size={expected_size}, remote size={remote_size})."
        )


def upload_lossless(
    client: OpenAI,
    path: Path,
    *,
    filename: str,
    mime_type: str,
    upload_cache: dict[tuple[str, int, str], str],
) -> str:
    local_sha256, local_size = compute_sha256_and_size(path)
    cache_key = (local_sha256, local_size, mime_type)
    cached_file_id = upload_cache.get(cache_key)
    if cached_file_id is not None:
        return cached_file_id

    uploaded = create_uploaded_file(
        client,
        path,
        filename=filename,
        mime_type=mime_type,
        purpose=PRIMARY_UPLOAD_PURPOSE,
    )

    if getattr(uploaded, "bytes", None) != local_size:
        raise ValueError(
            f"Upload size mismatch for {filename} "
            f"(local size={local_size}, uploaded size={getattr(uploaded, 'bytes', 'unknown')})."
        )

    try:
        verify_uploaded_content(
            client,
            file_id=uploaded.id,
            filename=filename,
            expected_sha256=local_sha256,
            expected_size=local_size,
        )
    except Exception as exc:
        if is_content_download_forbidden(exc):
            print(
                f"\nAttachment note: purpose '{PRIMARY_UPLOAD_PURPOSE}' does not allow content readback; "
                "continuing with upload-size verification only.\n"
            )
        else:
            raise

    upload_cache[cache_key] = uploaded.id
    return uploaded.id


def resolve_attachment_item(
    client: OpenAI,
    attachment: AttachmentSpec,
    *,
    upload_cache: dict[tuple[str, int, str], str],
) -> dict[str, str]:
    if attachment.kind == "image":
        if is_http_url(attachment.source):
            return {
                "type": "input_image",
                "image_url": attachment.source,
                "detail": "high",
            }

        file_path = resolve_local_path(attachment.source)
        ensure_readable_file(file_path)
        filename = file_path.name
        mime_type = guess_mime_type(file_path)
        if not mime_type.startswith("image/"):
            raise ValueError(f"/image requires an image file, got '{mime_type}': {file_path}")

        file_id = upload_lossless(
            client,
            file_path,
            filename=filename,
            mime_type=mime_type,
            upload_cache=upload_cache,
        )

        return {
            "type": "input_image",
            "file_id": file_id,
            "detail": "high",
        }

    if attachment.kind == "file":
        file_path = resolve_local_path(attachment.source)
        ensure_readable_file(file_path)
        filename = file_path.name
        mime_type = guess_mime_type(file_path)

        file_id = upload_lossless(
            client,
            file_path,
            filename=filename,
            mime_type=mime_type,
            upload_cache=upload_cache,
        )

        return {
            "type": "input_file",
            "file_id": file_id,
            "filename": filename,
        }

    raise ValueError(f"Unsupported attachment kind: {attachment.kind}")


def build_input_payload(
    client: OpenAI,
    parts: list[str | AttachmentSpec],
    *,
    upload_cache: dict[tuple[str, int, str], str],
):
    has_attachment = any(isinstance(part, AttachmentSpec) for part in parts)
    if not has_attachment:
        return "".join(part for part in parts if isinstance(part, str))

    content: list[dict[str, str]] = []
    for part in parts:
        if isinstance(part, AttachmentSpec):
            content.append(
                resolve_attachment_item(
                    client,
                    part,
                    upload_cache=upload_cache,
                )
            )
        elif part:
            text = part.strip()
            if text:
                content.append({"type": "input_text", "text": text})

    if not content:
        return ""
    return [{"role": "user", "content": content}]

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
    upload_cache: dict[tuple[str, int, str], str] = {}
    
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
            raw_input = session.prompt(message="\nUser:\n")
            if raw_input == "":
                continue

            parsed_turn = parse_turn_input(raw_input)
            if parsed_turn.show_help:
                print_slash_help()
                continue

            if not parsed_turn.parts:
                continue

            try:
                prepared_input = build_input_payload(
                    client,
                    parsed_turn.parts,
                    upload_cache=upload_cache,
                )
            except Exception as exc:
                print(f"\nAttachment error: {exc}\n")
                continue

            if not prepared_input:
                continue

            def make_stream():
                if response_id is None:
                    return client.responses.stream(
                        model="gpt-5.2",
                        tools=tools,
                        input=prepared_input,
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
