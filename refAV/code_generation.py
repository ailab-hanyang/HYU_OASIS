from pathlib import Path
import os
import ast
import sys
from pathlib import Path
from typing import List, Optional, TextIO
import json
import time
import re
import refAV.paths as paths

# API specific imports located within LLM-specific scenario prediction functions


def normalize_output_scenario_calls(code_block: str) -> str:
    """
    Normalizes output_scenario function calls to ensure they follow the format:
    output_scenario(<custom_variable_name>, description, log_dir, output_dir)
    """
    # Pattern to match output_scenario function calls
    # This handles multi-line function calls and various parameter formats
    pattern = r"output_scenario\s*\(\s*([^)]+)\s*\)"

    def normalize_call(match):
        params_str = match.group(1)

        # Split parameters more carefully, handling nested parentheses and quotes
        params = parse_function_parameters(params_str)

        if not params:
            return match.group(0)  # Return original if parsing fails

        # Initialize variables for the normalized format
        custom_variable = None
        description_param = "description"
        log_dir_param = "log_dir"
        output_dir_param = "output_dir"

        # Process parameters
        processed_params = []
        for i, param in enumerate(params):
            param = param.strip()

            # Handle keyword arguments
            if "=" in param:
                key, value = param.split("=", 1)
                key = key.strip()
                value = value.strip()

                if key == "scenario":
                    # If scenario= is used, treat the value as the custom variable
                    custom_variable = value
            else:
                # Handle positional arguments
                if i == 0:
                    custom_variable = param

        # If no custom variable was found, use the first parameter
        if custom_variable is None and params:
            custom_variable = params[0].strip()

        # Construct the normalized call
        normalized_call = f"output_scenario(\n    {custom_variable},\n    {description_param},\n    {log_dir_param},\n    {output_dir_param}\n)"

        return normalized_call

    # Apply the normalization
    normalized_code = re.sub(pattern, normalize_call, code_block, flags=re.DOTALL)

    return normalized_code


def parse_function_parameters(params_str: str) -> list[str]:
    """
    Parse function parameters, handling nested parentheses, quotes, and commas properly.
    """
    params = []
    current_param = ""
    paren_depth = 0
    in_quotes = False
    quote_char = None

    i = 0
    while i < len(params_str):
        char = params_str[i]

        # Handle escape sequences
        if char == "\\" and i + 1 < len(params_str):
            current_param += char + params_str[i + 1]
            i += 2
            continue

        # Handle quotes
        if char in ['"', "'"]:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None

        # Handle parentheses (only when not in quotes)
        elif not in_quotes:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "," and paren_depth == 0:
                # This comma separates parameters
                params.append(current_param.strip())
                current_param = ""
                i += 1
                continue

        current_param += char
        i += 1

    # Add the last parameter
    if current_param.strip():
        params.append(current_param.strip())

    return params


def extract_and_save_code_blocks(
    message, description=None, output_dir: Path = Path(".")
) -> list[Path]:
    """
    Extracts Python code blocks from a message and saves them to files based on their description variables.
    Handles both explicit Python code blocks (```python) and generic code blocks (```).
    """

    # Split the message into lines and handle escaped characters
    lines = message.replace("\\n", "\n").replace("\\'", "'").split("\n")
    in_code_block = False
    current_block = []
    code_blocks = []

    for line in lines:
        # Check for code block markers

        if line.strip().startswith("```"):
            # If we're not in a code block, start one
            if not in_code_block:
                in_code_block = True
                current_block = []
            # If we're in a code block, end it
            else:
                in_code_block = False
                if current_block:  # Only add non-empty blocks
                    code_blocks.append("\n".join(current_block))
                current_block = []
            continue

        # If we're in a code block, add the line
        if in_code_block:
            # Skip the "python" language identifier if it's there
            if line.strip().lower() == "python":
                continue
            if "description =" in line:
                continue

            current_block.append(line)

    # Process each code block
    filenames = []
    for i, code_block in enumerate(code_blocks):

        cleaned_code_block = normalize_output_scenario_calls(code_block)
        # Save the code block
        if description:
            filename = output_dir / f"{description}.txt"
        else:
            filename = output_dir / "default.txt"

        try:
            with open(filename, "w") as f:
                f.write(cleaned_code_block)
            filenames.append(filename)
        except Exception as e:
            print(f"Error saving file {filename}: {e}")

    return filenames


def build_context(context_path=paths.PROMPT_DIR / "RefAV") -> str:
    # Layout under context_path:
    #   shared/  - atomic_functions.txt, categories.txt, examples.txt (used by
    #              both the single-agent and multi-agent pipelines)
    #   single/  - system_prompt.txt (single-agent only — read here)
    #   multi/   - deconstructor/selector/coder prompts (read by multi_agent.py)
    shared = context_path / "shared"
    single = context_path / "single"

    with open(shared / "atomic_functions.txt", "r") as f:
        refav_context = f.read()
    with open(shared / "categories.txt", "r") as f:
        av2_categories = f.read()
    with open(shared / "examples.txt", "r") as f:
        prediction_examples = f.read()
    with open(single / "system_prompt.txt", "r") as f:
        context = f.read()

    # Brace-safe placeholder injection — atomic_functions.txt docstrings may
    # contain literal `{key: val}` snippets (e.g. nth_object_in_direction
    # returns "{ track_uuid: { related_uuid: [ts...] } }"). str.format() would
    # misread them as placeholders. Plain string replace avoids that entirely.
    for placeholder, value in (
        ("{refav_context}", refav_context),
        ("{av2_categories}", av2_categories),
        ("{prediction_examples}", prediction_examples),
    ):
        context = context.replace(placeholder, value)

    return context


def predict_scenario_from_description(
    natural_language_description,
    output_dir: Path,
    custom_context=None,
    model_name: str = "gemini-2.0-flash",
    local_model=None,
    local_tokenizer=None,
    destructive=False,
    multi_agent: bool = False,
    split: str = "val",
) -> Path:
    """
    Generates a scenario definition (a code snippet utilizing the atomic functions) from the given description.

    Args:
        custom_prompt: A string describing how to define a scenario. It must be in the form "context {natural_language_description} context".
            It is recommended to use build_prompt() to make this string.
    """
    # Multi-agent path: Selector + Coder via refAV/multi_agent.py.
    # Falls back to single-agent on Selector failure (handled below).
    if multi_agent:
        from refAV.multi_agent import predict_scenario_multi_agent
        def _llm_call(prompt: str, model_name_: str) -> str:
            if "claude" in model_name_.lower():
                return predict_scenario_anthropic(prompt, model_name_)
            if "gemini" in model_name_.lower():
                return predict_scenario_google(prompt, model_name_)
            if "gpt" in model_name_.lower():
                return predict_scenario_openai(prompt, model_name_)
            if "qwen" in model_name_.lower():
                return predict_scenario_qwen(prompt, local_model, local_tokenizer)
            raise ValueError(f"Unknown model_name for multi-agent: {model_name_}")
        ma_path = predict_scenario_multi_agent(
            natural_language_description, output_dir, model_name, _llm_call,
            split=split,
        )
        if ma_path is not None:
            return ma_path
        # Selector failure -> fall through to single-agent below (multi_agent=False semantics)

    output_dir = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    definition_filename = output_dir / (natural_language_description + ".txt")

    if definition_filename.exists() and not destructive:
        print(
            f"Cached scenario for description {natural_language_description} already found."
        )
        return definition_filename

    if not custom_context:
        custom_context = build_context()

    # The build_context() result has already been through one str.format() pass, so the literal
    # braces from atomic_functions.txt (docstring examples like `{ track_uuid: {...} }`) remain
    # as single braces. Calling .format() again here would misinterpret those single braces as
    # placeholders and raise a KeyError (e.g. ' track_uuid'). Since the only placeholder we need
    # to fill is one ({natural_language_description}), using a simple .replace() instead of format
    # avoids any conflict with the other braces.
    prompt = custom_context.replace(
        "{natural_language_description}", natural_language_description
    )

    if "gemini" in model_name.lower():
        response = predict_scenario_google(prompt, model_name)
    elif "gpt" in model_name.lower():
        response = predict_scenario_openai(prompt, model_name)
    elif "qwen" in model_name.lower():
        response = predict_scenario_qwen(prompt, local_model, local_tokenizer)
    elif "claude" in model_name.lower():
        response = predict_scenario_anthropic(prompt, model_name)

    try:
        definition_filename = extract_and_save_code_blocks(
            response, output_dir=output_dir, description=natural_language_description
        )[-1]
        print(f"{natural_language_description} definition saved to {output_dir}")
        return definition_filename
    except Exception as e:
        print(e)
        print(response)
        print(f"Error saving description {natural_language_description}")
        return


def predict_scenario_google(prompt, model_name):
    from google import genai

    """
    Available models:
    gemini-2.5-flash-preview-04-17
    gemini-2.0-flash
    """

    time.sleep(6)  # Free API limited to 10 requests per minute
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    config = {
        "temperature": 0.8,
        "max_output_tokens": 4096,
    }

    response = client.models.generate_content(
        model=model_name, contents=prompt, config=config
    )

    return response.text


def _resolve_anthropic_model_id(name: str) -> str:
    """The `LLM:` field in experiments.yml serves as both the .txt cache directory name and the
    API model ID. When an experiment variant has a long suffix appended (e.g.
    `claude-sonnet-4-6-260519-ego-L1v2-p6-yawfix-...-reversing`),
    sending it as-is to the Anthropic API results in a 404 NotFoundError. We keep the directory
    name unchanged and extract only the canonical model ID right before the API call.

    Matching rules:
      1. `claude-N-M-(sonnet|opus|haiku)-YYYYMMDD`     — old dated naming
         e.g. claude-3-5-sonnet-20241022, claude-3-7-sonnet-20250219
      2. `claude-(sonnet|opus|haiku)-N-M(-YYYYMMDD)?`  — new naming
         e.g. claude-sonnet-4-6, claude-sonnet-4-5-20250929, claude-haiku-4-5-20251001

    Return only up to the above prefix. If there is no match, return the input as-is (— any error
    will be reported by the API).
    """
    import re
    # 1) old (digit-major-minor-family-date)
    m = re.match(r'^(claude-\d-\d-(?:sonnet|opus|haiku)-\d{8})', name)
    if m:
        return m.group(1)
    # 2) new (family-major-minor with optional date)
    m = re.match(r'^(claude-(?:sonnet|opus|haiku)-\d-\d(?:-\d{8})?)', name)
    if m:
        return m.group(1)
    return name


def predict_scenario_anthropic(prompt, model_name):
    import anthropic
    import time

    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        # api_key="my_api_key",
    )

    # Strip off the variant suffix (.txt directory name) and pass only the canonical model ID to the API.
    api_model = _resolve_anthropic_model_id(model_name)
    if api_model != model_name:
        print(f"[anthropic] mapped LLM dir name → API model: {model_name!r} → {api_model!r}")

    # For reasoning-family models such as Opus 4.7/4.8, `temperature` is deprecated —
    # the API rejects it with a 400 BadRequest (observed: "`temperature` is deprecated for this model.").
    # Earlier Opus (4.6 and below), Sonnet, Haiku, and old Claude models accept temperature, so branch on the model name.
    create_kwargs = {
        "model":      api_model,
        "max_tokens": 4096,
        "messages":   [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    # For known deprecated models, omit temperature from the start to avoid a wasted call (400 → retry).
    # (Even if a model not on the list rejects it, the BadRequestError handler below is the final safety net.)
    TEMPERATURE_DEPRECATED = ("opus-4-7", "opus-4-8")
    if not any(tag in api_model for tag in TEMPERATURE_DEPRECATED):
        # Sonnet / Haiku / earlier Opus / old Claude accept temperature.
        # [FIX] temperature 0.5 -> 0.0 to make code generation deterministic
        create_kwargs["temperature"] = 0.0
        
    # On 429 (RateLimitError) / 529 (overloaded), retry with exponential backoff.
    # We frequently hit the org-level token-per-minute limit (2M tpm) — since the window resets after 60s,
    # wait at least 60s. Multiply by ×1.7 on each retry, capped at 600s.
    max_retries = 12
    base_delay = 60.0
    for attempt in range(max_retries + 1):
        try:
            message = client.messages.create(**create_kwargs)
            break
        except anthropic.BadRequestError as e:
            # Cases like `temperature is deprecated for this model` — remove temperature and retry.
            # (If a BadRequest still occurs after it has already been removed, it's a different issue → raise)
            if "temperature" in create_kwargs and "temperature" in str(getattr(e, "message", e)):
                create_kwargs.pop("temperature", None)
                print(f"[anthropic 400] temperature deprecated — model={api_model!r}, removing and retrying")
                continue
            raise
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status not in (429, 529) or attempt == max_retries:
                raise
            # Honor server's retry-after header if present, else exponential backoff.
            retry_after_hdr = None
            resp = getattr(e, "response", None)
            if resp is not None:
                retry_after_hdr = resp.headers.get("retry-after") or resp.headers.get("anthropic-ratelimit-input-tokens-reset")
            try:
                wait = float(retry_after_hdr) if retry_after_hdr else base_delay * (1.7 ** attempt)
            except (TypeError, ValueError):
                wait = base_delay * (1.7 ** attempt)
            wait = max(60.0, min(wait, 600.0))
            print(f"[anthropic {status}] retry {attempt+1}/{max_retries} after {wait:.0f}s — {type(e).__name__}")
            time.sleep(wait)

    # Convert the message content to string
    if hasattr(message, "content"):
        content = message.content
    else:
        raise ValueError("Message object doesn't have 'content' attribute")

    if hasattr(content[0], "text"):
        text_response = content[0].text
    elif isinstance(content, list):
        text_response = "\n".join(str(item) for item in content)
    else:
        text_response = str(content)

    return text_response


def predict_scenario_openai(prompt, model_name):
    import openai

    client = openai.OpenAI()

    response = client.responses.create(
        model=model_name,
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    )

    text_response = response.output_text

    return text_response


def process_batch_prompts_claude(prompts, prompt_ids, model_name):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()

    message_batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id=prompt_id,
                params=MessageCreateParamsNonStreaming(
                    model=model_name,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                ),
            )
            for prompt, prompt_id in zip(prompts, prompt_ids)
        ]
    )

    print(message_batch)

    pass


def load_qwen(model_name="Qwen2.5-7B-Instruct"):

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    qwen_model_name = "Qwen/" + model_name
    model = AutoModelForCausalLM.from_pretrained(
        qwen_model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)

    return model, tokenizer


def predict_scenario_qwen(prompt, model=None, tokenizer=None):

    if model == None or tokenizer == None:
        model, tokenizer = load_qwen()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=2048)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


class FunctionInfo:
    """Holds extracted information for a single function."""

    def __init__(
        self,
        name: str,
        signature_lines: List[str],
        docstring: Optional[str],
        col_offset: int,
    ):
        self.name = name
        # Keep signature as lines to preserve original formatting/indentation
        self.signature_lines = signature_lines
        self.docstring = docstring
        self.col_offset = col_offset  # Store the column offset of the 'def' keyword

    def format_for_output(self) -> str:
        """Formats the function signature and docstring for display, including triple quotes."""
        # Determine base indentation from the 'def' line's column offset
        base_indent = " " * self.col_offset
        # Assume standard 4-space indentation for the body/docstring relative to the 'def' line
        body_indent = base_indent + "    "

        # Start with the signature lines
        # Strip trailing whitespace but keep leading whitespace (which is the base_indent)
        output_lines = [line.rstrip() for line in self.signature_lines]

        if self.docstring is not None:
            # Split the raw docstring content by lines
            docstring_lines = self.docstring.splitlines()

            # Add opening quotes line indented by body_indent
            output_lines.append(f'{body_indent}"""')

            # Add the docstring content lines, each indented by body_indent
            # ast.get_docstring already removes the *minimal* indentation from the *content block*.
            # So we just need to add the *body indent* to each line of the processed content.
            for line in docstring_lines:
                output_lines.append(f"{body_indent}{line}")

            # Add closing quotes line indented by body_indent
            output_lines.append(f'{body_indent}"""')

        # Join the lines
        return "\n".join(output_lines).strip()

# --- AST Visitor to extract Function Info ---


class FunctionDocstringExtractor(ast.NodeVisitor):
    """AST visitor to find function definitions and extract their info."""

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        # Update the type hint for extracted_info to reflect the modified FunctionInfo
        self.extracted_info: List[FunctionInfo] = []

    def visit_FunctionDef(self, node):
        """Visits function definitions (def)."""
        name = node.name

        # Get the docstring using the standard ast helper
        docstring_content = ast.get_docstring(node)

        # Get the column offset of the 'def' keyword
        col_offset = node.col_offset

        # Determine the line number where the function body actually starts.
        body_start_lineno = node.lineno + 1
        if node.body:
            first_body_node = node.body[0]
            body_start_lineno = first_body_node.lineno

        # Extract signature lines: from the line of 'def' up to the line before the body starts.
        signature_lines_raw = self.source_lines[node.lineno - 1 : body_start_lineno - 1]

        # Pass the col_offset when creating the FunctionInfo object
        self.extracted_info.append(
            FunctionInfo(name, signature_lines_raw, docstring_content, col_offset)
        )

        # We still don't generically visit children unless you uncomment generic_visit
        # self.generic_visit(node) # Keep commented unless you need nested functions/classes

    def visit_AsyncFunctionDef(self, node):
        """Visits async function definitions (async def)."""
        # Call the same logic as visit_FunctionDef
        self.visit_FunctionDef(node)


# --- Main Parsing Function ---


def parse_python_functions_with_docstrings(
    file_path: Path,
    output_path: Path = paths.PROMPT_DIR / "RefAV/shared/atomic_functions.txt",
) -> List[FunctionInfo]:
    """
    Parses a Python file to extract function definitions (signature) and their docstrings,
    excluding decorators.

    Args:
        file_path: Path to the Python file.

    Returns:
        A list of FunctionInfo objects, each containing the function name,
        signature lines (without decorators), and docstring. Returns an empty
        list in case of errors.
    """
    try:
        # Read the file content, specifying encoding for robustness
        source_code = file_path.read_text(encoding="utf-8")
        # Keep original lines to reconstruct signatures
        lines = source_code.splitlines()

        # Parse the source code into an Abstract Syntax Tree
        tree = ast.parse(source_code)

        # Use the visitor to walk the tree and extract info
        visitor = FunctionDocstringExtractor(lines)
        visitor.visit(tree)  # Start the traversal

        with open(output_path, "w") as file:
            display_function_info(visitor.extracted_info, file)

        return visitor.extracted_info

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}", file=sys.stderr)
        return []


def display_function_info(
    function_info_list: List[FunctionInfo], output_stream: TextIO = sys.stdout
):
    """
    Displays the extracted function information (signature and docstring)
    to the specified output stream in the requested text format.

    Args:
        function_info_list: A list of FunctionInfo objects.
        output_stream: The stream to write the output to (e.g., sys.stdout, a file object).
    """
    for i, func_info in enumerate(function_info_list):
        if i > 0:
            # Add a separator between function outputs for clarity, matching the previous output
            output_stream.write("\n\n")

        # Use the format_for_output method to get the combined signature and docstring
        formatted_text = func_info.format_for_output()
        output_stream.write(formatted_text)
        output_stream.write("\n")  # Ensure a newline after each function block

def generate_all_scenario_definitions(split:str, model_name:str):

    atomic_functions_path = Path("refAV/atomic_functions.py")
    parse_python_functions_with_docstrings(atomic_functions_path)

    all_descriptions = set()
    with open(
        paths.SM_DOWNLOAD_DIR / f"log_prompt_pairs_{split}.json", "rb"
    ) as file:
        log_prompt_pairs = json.load(file)

    for prompts in log_prompt_pairs.values():
        all_descriptions.update(prompts)

    print(len(all_descriptions))

    output_dir = paths.LLM_PRED_DIR / "RefAV"
    context = build_context(paths.PROMPT_DIR / "RefAV")

    for description in all_descriptions:
        predict_scenario_from_description(
            description,
            output_dir,
            model_name=model_name,
            custom_context=context,
        )

if __name__ == "__main__":

    atomic_functions_path = Path("refAV/atomic_functions.py")
    parse_python_functions_with_docstrings(atomic_functions_path)

    all_descriptions = set()
    with open(
        paths.SM_DOWNLOAD_DIR / "log_prompt_pairs_val.json", "rb"
    ) as file:
        lpp_val = json.load(file)

    # with open('av2_sm_downloads/log_prompt_pairs_test.json', 'rb') as file:
    #    lpp_test = json.load(file)

    for log_id, prompts in lpp_val.items():
        all_descriptions.update(prompts)
    # for log_id, prompts in lpp_test.items():
    #    all_descriptions.update(prompts)

    print(len(all_descriptions))

    model_name = "claude-3-7-sonnet-20250219"
    output_dir = paths.LLM_PRED_DIR / "RefAV"
    nuprompt_context = build_context(paths.PROMPT_DIR / "RefAV")

    for description in all_descriptions:
        predict_scenario_from_description(
            description,
            output_dir,
            model_name=model_name,
            custom_context=nuprompt_context,
        )
