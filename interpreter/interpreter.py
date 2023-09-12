"""
open_interpreter.py

Main module for the Open Interpreter assistant. Handles chatting with AI
models like GPT-3, Code-Llama, etc. and executing code.
"""
import os
import time
import traceback
import json
import platform
import readline
import getpass
import pkg_resources

from rich.rule import Rule
from rich import print
from rich.markdown import Markdown
import tokentrim as tt
import litellm
import requests

from .cli import cli
from .utils import merge_deltas, parse_partial_json
from .message_block import MessageBlock
from .code_block import CodeBlock
from .code_interpreter import CodeInterpreter
from .get_hf_llm import get_hf_llm

# Function schema for gpt-4
function_schema = {
  "name": "run_code",
  "description": "Executes code on the user's machine and returns the output",
  "parameters": {
    "type": "object",
    "properties": {
      "language": {
        "type": "string",
        "description":
        "The programming language",
        "enum": ["python", "shell", "applescript", "javascript", "html"]
      },
      "code": {
        "type": "string",
        "description": "The code to execute"
      }
    },
    "required": ["language", "code"]
  },
}

# Message for when users don't have an OpenAI API key.
MISSING_API_KEY_MESSAGE = """> OpenAI API key not found

To use `GPT-4` (recommended) please provide an OpenAI API key.

To use `Code-Llama` (free but less capable) press `enter`.
"""

# Message for when users don't have an OpenAI API key.
MISSING_AZURE_INFO_MESSAGE = """> Azure OpenAI Service API info not found

To use `GPT-4` (recommended) please provide an Azure OpenAI API key,
a API base, a deployment name and a API version.

To use `Code-Llama` (free but less capable) press `enter`.
"""

CONFIRM_MODE_MESSAGE = """
**Open Interpreter** will require approval before running code.
Use `interpreter -y` to bypass this.

Press `CTRL-C` to exit.
"""


class Interpreter:
    """
    Main Interpreter class that handles chatting with the AI assistant.

    Attributes:
        messages (list): Message history with the assistant.
        temperature (float): Sampling temperature for the AI assistant.
        api_key (str): OpenAI API key.
        auto_run (bool): Whether to automatically run code without prompting.
        local (bool): Whether to run locally with Code-Llama vs OpenAI API.
        model (str): Which AI model to use (gpt-4, code-llama, etc).
        debug_mode (bool): Enable debug logging and info.
        use_azure (bool): Whether to use Azure OpenAI service.
        azure_api_base (str): Azure API base URL.
        azure_api_version (str): Azure API version.
        azure_deployment_name (str): Azure deployment name.
        system_message (str): Default system prompt message for the AI.
        code_interpreters (dict): Code interpreters for executing code.
        active_block (Block): Currently active message or code block.
        llama_instance (LLM): Instance of locally running Code-Llama.

    Methods:
        cli(): Parses command line args and starts the chat loop.
        get_info_for_system_message(): Gets info to add to system message.
        reset(): Reset the message history.
        load(): Load a message history to use.
        chat(): Main chat loop with the AI assistant.
        verify_api_key(): Checks for and loads API key if needed.
        end_active_block(): Ends the currently active message/code block.
        respond(): Gets AI response and handles code execution.
    """
    def __init__(self):
        self.messages = []
        self.temperature = 0.001
        self.api_key = None
        self.auto_run = False
        self.local = False
        self.model = "gpt-4"
        self.debug_mode = False
        self.api_base = None
        self.context_window = 2000
        self.max_tokens = 750
        # Azure OpenAI
        self.use_azure = False
        self.azure_api_base = None
        self.azure_api_version = None
        self.azure_deployment_name = None
        self.azure_api_type = "azure"

        # Get default system message
        here = os.path.abspath(os.path.dirname(__file__))
        with open(
            os.path.join(
                here,
                'system_message.txt'),
            'r',
                encoding="utf8") as file_path:
            self.system_message = file_path.read().strip()

        # Store Code Interpreter instances for each language
        self.code_interpreters = {}

        # No active block to start
        # (blocks are visual representation of messages on the terminal)
        self.active_block = None

        # Note: While Open Interpreter can use Llama, we will prioritize gpt-4.
        # gpt-4 is faster, smarter, can call functions,
        # and is all-around easier to use.
        self.llama_instance = None

    def log_to_api(self, message):
        """
        Logs a message to a provided API endpoint.

        Args:
            message (dict): The message to log.
        """
        api_endpoint = 'https://aoai-eastus2-airesearch.openai.azure.com'

        # Convert the message to JSON format
        message_json = json.dumps(message)

        # Send a POST request to your API
        try:
            response = requests.post(
                api_endpoint,
                data=message_json,
                timeout=15)
            if response.status_code == 200:
                print("Message successfully logged to API.")
            else:
                print(
                  "Failed to log message to API. Status code:",
                  response.status_code)
        except Exception as ex_err:
            print("Error logging message to API:", str(ex_err))

    def cli(self):
        """
        Parses command line arguments and starts the interactive chat loop.

        Modifies the Interpreter instance according to any command line flags.
        """
        # The cli takes the current instance of Interpreter,
        # modifies it according to command line flags, then runs chat.
        cli(self)

    def get_info_for_system_message(self):
        """
        Gets relevent information for the system message.
        """

        info = ""

        # Add user info
        username = getpass.getuser()
        current_working_directory = os.getcwd()
        operating_system = platform.system()

        info += f"[User Info]\nName: {username}"
        info += f"\nCWD: {current_working_directory}\n"
        info += f"OS: {operating_system}"

        if not self.local:

            # Open Procedures is an open-source database of tiny,
            # structured coding tutorials.
            # We can query it semantically and append relevant
            # tutorials/procedures to our system message:

            # Use the last two messages' content or function call to
            # semantically search
            query = []
            for message in self.messages[-2:]:
                message_for_semantic_search = {"role": message["role"]}
                if "content" in message:
                    message_for_semantic_search["content"] = message["content"]
                if "function_call" in message \
                    and "parsed_arguments" \
                        in message["function_call"]:
                    message_for_semantic_search["function_call"] = \
                        message["function_call"]["parsed_arguments"]
                query.append(message_for_semantic_search)

            # Use them to query Open Procedures
            url = "https://open-procedures.replit.app/search/"

            try:
                relevant_procedures = requests.get(
                  url,
                  data=json.dumps(query),
                  timeout=30).json()["procedures"]
                info += "\n\n# Recommended Procedures\n" + \
                    "\n---\n".join(relevant_procedures) + \
                    "\nIn your plan, include steps and, if present, " \
                    "*EXACT CODE SNIPPETS**" \
                    "(especially for deprecation notices, " \
                    "**WRITE THEM INTO YOUR PLAN -- "\
                    "underneath each numbered step** as they will VANISH "\
                    "once you execute your first line of code, " \
                    "so WRITE THEM DOWN NOW if you need them) " \
                    "from the above procedures if they are relevant " \
                    "to the task. Again, include **VERBATIM CODE SNIPPETS**" \
                    "from the procedures above if they are relevant " \
                    "to the task **directly in your plan.**"
            except Exception as sysmsg_info_err:
                # For someone, this failed for a super secure SSL reason.
                raise sysmsg_info_err

        elif self.local:
            # Tell Code-Llama how to run code.
            info += "\n\nTo run code, write a fenced code block " \
                "(i.e ```python or ```shell) in markdown. When you close it" \
                " with ```, it will be run. You'll then be given its output."
            # We make references in system_message.txt to the "function"
            # it can call, "run_code".

        return info

    def reset(self):
        """
        Resets the message history.
        """
        self.messages = []
        self.code_interpreters = {}

    def load(self, messages):
        """
        Loads a provided message history.

        Args:
            messages (list): Message history to load.
        """
        self.messages = messages

    def chat(self, message=None, return_messages=False):
        """
        Main chat loop with the AI assistant.

        Either chats interactively or responds to a single message.

        Args:
            message (str, optional):
            A single message to respond to.
            return_messages (bool, optional):
            Whether to return the message history.

        Returns:
            list: The message history if return_messages is True.
        """

        # Connect to an LLM (an large language model)
        if not self.local:
            # gpt-4
            self.verify_api_key()

        # ^ verify_api_key may set self.local to True,
        # so we run this as an 'if', not 'elif':
        if self.local:
            self.model = "code-llama"

            # Code-Llama
            if self.llama_instance is None:

                # Find or install Code-Llama
                try:
                    self.llama_instance = get_hf_llm(
                        self.model,
                        self.debug_mode,
                        self.context_window)
                    if self.llama_instance is None:
                        # They cancelled.
                        return
                except Exception as err_chat:
                    traceback.print_exc(err_chat)
                    # If it didn't work, apologize and switch to GPT-4
                    print(Markdown("".join([
                      "> Failed to install `Code-LLama`.",
                      "\n\n**We have likely not built the proper `Code-Llama` "
                      " support for your system.**",
                      "\n\n*( Running language models locally is a difficult "
                      "task! * If you have insight into the best way to "
                      "implement this across platforms/architectures, please "
                      "join the Open Interpreter community Discord and "
                      "consider contributing the project's development. )",
                      "\n\nPlease press enter to switch to `GPT-4` "
                      "(recommended)."
                    ])))
                    input()

                # Switch to GPT-4
                self.local = False
                self.model = "gpt-4"
                self.verify_api_key()

        # Display welcome message
        welcome_message = ""

        if self.debug_mode:
            welcome_message += "> Entered debug mode"

        # If self.local, we actually don't use self.model
        # (self.auto_run is like advanced usage, we display no messages)
        if not self.local and not self.auto_run:
            welcome_message += f"\n> Model set to `{self.model.upper()}`\n\n" \
              "**Tip:** To run locally, use `interpreter --local`"

        if self.local:
            welcome_message += "\n> Model set to `Code-Llama`"
        # If not auto_run, tell the user we'll ask permission to run code
        # We also tell them here how to exit Open Interpreter
        if not self.auto_run:
            welcome_message += "\n\n" + CONFIRM_MODE_MESSAGE

        welcome_message = welcome_message.strip()

        # Print welcome message with newlines on either side (aesthetic choice)
        # unless we're starting with a blockquote (aesthetic choice)
        if welcome_message != "":
            if welcome_message.startswith(">"):
                print(Markdown(welcome_message), '')
            else:
                print('', Markdown(welcome_message), '')

        # Check if `message` was passed in by user
        if message:
            # If it was, we respond non-interactivley
            self.messages.append({"role": "user", "content": message})
            self.respond()

        else:
            # If it wasn't, we start an interactive chat
            while True:
                try:
                    user_input = input("> ").strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print()  # Aesthetic choice
                    break

                # Use `readline` to let users up-arrow
                # to previous user messages,
                # which is a common behavior in terminals.
                readline.add_history(user_input)

                # Add the user message to self.messages
                self.messages.append(
                    {
                        "role": "user",
                        "content": user_input
                    })

                # Let the user turn on debug mode mid-chat
                if user_input == "%debug":
                    print('', Markdown("> Entered debug mode"), '')
                    print(self.messages)
                    self.debug_mode = True
                    continue

            # Respond, but gracefully handle CTRL-C / KeyboardInterrupt
            try:
                self.respond()
            except KeyboardInterrupt:
                pass
            finally:
                # Always end the active block.
                # Multiple Live displays = issues
                self.end_active_block()

            if return_messages:
                return self.messages

    def verify_api_key(self):
        """
        Makes sure we have an AZURE_API_KEY or OPENAI_API_KEY.
        """
        if self.use_azure:
            all_env_available = (
                ('AZURE_API_KEY' in os.environ
                    or 'OPENAI_API_KEY' in os.environ)
                and 'AZURE_API_BASE' in os.environ
                and 'AZURE_API_VERSION' in os.environ
                and 'AZURE_DEPLOYMENT_NAME' in os.environ
                                )
            if all_env_available:
                self.api_key = \
                    os.environ.get('AZURE_API_KEY') or \
                    os.environ['OPENAI_API_KEY']
                self.azure_api_base = \
                    os.environ['AZURE_API_BASE']
                self.azure_api_version = \
                    os.environ['AZURE_API_VERSION']
                self.azure_deployment_name = \
                    os.environ['AZURE_DEPLOYMENT_NAME']
                self.azure_api_type = \
                    os.environ.get('AZURE_API_TYPE', 'azure')
            else:
                # This is probably their first time here!
                self._print_welcome_message()
                time.sleep(1)

                print(Rule(style="white"))

                print(
                    Markdown(
                        MISSING_AZURE_INFO_MESSAGE
                        ), '', Rule(
                            style="white"
                            ), '')
                response = input("Azure OpenAI API key: ")

            if response == "":
                # User pressed `enter`, requesting Code-Llama

                print(Markdown(
                    "> Switching to `Code-Llama`...\n\n"
                    "**Tip:** Run `interpreter --local` "
                    "to automatically use `Code-Llama`."),
                        ''
                    )
                time.sleep(2)
                print(Rule(style="white"))

                # Temporarily, for backwards (behavioral) compatability,
                # we've moved this part of llama_2.py here.
                import inquirer

                print(
                    '',
                    Markdown(
                        "**Open Interpreter** will use `Code Llama` "
                        "for local execution. "
                        "Use your arrow keys to set up the model."),
                    '')

                models = {
                    '7B': 'TheBloke/CodeLlama-7B-Instruct-GGUF',
                    '13B': 'TheBloke/CodeLlama-13B-Instruct-GGUF',
                    '34B': 'TheBloke/CodeLlama-34B-Instruct-GGUF'
                }

                parameter_choices = list(models.keys())
                questions = [
                    inquirer.List(
                        'param',
                        message="Parameter count (smaller is faster, "
                        "larger is more capable)",
                        choices=parameter_choices)]
                answers = inquirer.prompt(questions)
                chosen_param = answers['param']

                self.model = models[chosen_param]
                self.local = True

                return

            else:
                self.api_key = response
                self.azure_api_base = input("Azure OpenAI API base: ")
                self.azure_deployment_name = input(
                    "Azure OpenAI deployment name of GPT: ")
                self.azure_api_version = input("Azure OpenAI API version: ")
                print('', Markdown(
                    "**Tip:** To save this key for later, "
                    "run `export AZURE_API_KEY=your_api_key "
                    "AZURE_API_BASE=your_api_base "
                    "AZURE_API_VERSION=your_api_version "
                    "AZURE_DEPLOYMENT_NAME=your_gpt_deployment_name` "
                    "on Mac/Linux or "
                    "`setx AZURE_API_KEY your_api_key "
                    "AZURE_API_BASE your_api_base "
                    "AZURE_API_VERSION your_api_version "
                    "AZURE_DEPLOYMENT_NAME your_gpt_deployment_name` "
                    "on Windows."),
                        '')
                time.sleep(2)
                print(Rule(style="white"))

            litellm.api_type = self.azure_api_type
            litellm.api_base = self.azure_api_base
            litellm.api_version = self.azure_api_version
            litellm.api_key = self.api_key

        else:
            if self.api_key is None:
                if 'OPENAI_API_KEY' in os.environ:
                    self.api_key = os.environ['OPENAI_API_KEY']
            else:
                # This is probably their first time here!
                self._print_welcome_message()
                time.sleep(1)

                print(Rule(style="white"))

                print(
                    Markdown(MISSING_API_KEY_MESSAGE),
                    '',
                    Rule(style="white"),
                    '')
                response = input("OpenAI API key: ")

                if response == "":
                    # User pressed `enter`, requesting Code-Llama
                    self.local = True
                    print(
                        Markdown(
                            "> Switching to `Code-Llama`...\n\n"
                            "**Tip:** Run `interpreter --local` "
                            "to automatically use `Code-Llama`."
                            ), '')
                    time.sleep(2)
                    print(Rule(style="white"))

                    import inquirer

                    print('', Markdown(
                            "**Open Interpreter** will use `Code Llama` "
                            "for local execution. "
                            "Use your arrow keys to set up the model."), '')

                    models = {
                        '7B': 'TheBloke/CodeLlama-7B-Instruct-GGUF',
                        '13B': 'TheBloke/CodeLlama-13B-Instruct-GGUF',
                        '34B': 'TheBloke/CodeLlama-34B-Instruct-GGUF'
                    }

                    parameter_choices = list(models.keys())
                    questions = [inquirer.List(
                        'param',
                        message="Parameter count (smaller is faster, "
                        "larger is more capable)",
                        choices=parameter_choices)]
                    answers = inquirer.prompt(questions)
                    chosen_param = answers['param']

                    self.model = models[chosen_param]
                    self.local = True

                    return

                else:
                    self.api_key = response
                    print(
                        '',
                        Markdown(
                            "**Tip:** To save this key for later, "
                            "run `export OPENAI_API_KEY=your_api_key` "
                            "on Mac/Linux or "
                            "`setx OPENAI_API_KEY your_api_key` "
                            "on Windows."), '')
                    time.sleep(2)
                    print(Rule(style="white"))

            litellm.api_key = self.api_key
            if self.api_base:
                litellm.api_base = self.api_base

    def end_active_block(self):
        """
        Ends the currently active message or code block if one exists.
        """
        if self.active_block:
            self.active_block.end()
            self.active_block = None

    def respond(self):
        """
        Gets a response from the AI assistant and handles code execution.
        """
        # Add relevant info to system_message
        # (e.g. current working directory, username, os, etc.)
        info = self.get_info_for_system_message()

        if self.local:
            self.system_message = \
                "\n".join(self.system_message.split("\n")[:2])
            self.system_message += \
                "\nOnly do what the user asks you to do, " \
                "then ask what they'd like to do next."

        system_message = self.system_message + "\n\n" + info

        if self.local:
            messages = tt.trim(
                self.messages,
                max_tokens=(
                    self.context_window-self.max_tokens-25),
                system_message=system_message)
        else:
            messages = tt.trim(
                self.messages,
                self.model,
                system_message=system_message)

        if self.debug_mode:
            print("\n", "Sending `messages` to LLM:", "\n")
            print(messages)
            print()

        # Make LLM call
        if not self.local:
            # GPT

            error = ""

            for _ in range(3):  # 3 retries
                try:
                    if self.use_azure:
                        response = litellm.completion(
                            f"azure/{self.azure_deployment_name}",
                            messages=messages,
                            functions=[function_schema],
                            temperature=self.temperature,
                            stream=True,
                            )
                    else:
                        if self.api_base:
                            # The user set the api_base.
                            # litellm needs this to be "custom/{model}"
                            response = litellm.completion(
                                api_base=self.api_base,
                                model="custom/" + self.model,
                                messages=messages,
                                functions=[function_schema],
                                stream=True,
                                temperature=self.temperature,
                            )
                        else:
                            # Normal OpenAI call
                            response = litellm.completion(
                                model=self.model,
                                messages=messages,
                                functions=[function_schema],
                                stream=True,
                                temperature=self.temperature,
                            )

                    break
                except Exception as err_response:
                    if self.debug_mode:
                        traceback.print_exc(err_response)
                    error = traceback.format_exc()
                    time.sleep(3)
            else:
                raise Exception(error)

        elif self.local:
            # Code-Llama: Convert messages to prompt

            def messages_to_prompt(messages):

                for message in messages:
                    # Happens if it immediatly writes code
                    if "role" not in message:
                        message["role"] = "assistant"

                # Falcon prompt template
                if "falcon" in self.model.lower():
                    formatted_messages = ""
                    for message in messages:
                        formatted_messages += (
                            f"{message['role'].capitalize()}: "
                            f"{message['content']}\n"
                        )
                    formatted_messages = formatted_messages.strip()

                else:
                    # Llama prompt template
                    # Extracting the system prompt and initializing.
                    system_prompt = messages[0]['content']
                    formatted_messages = (
                        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
                        )

                    # Loop starting from the first user message
                    for index, item in enumerate(messages[1:]):
                        role = item['role']
                        content = item['content']

                        if role == 'user':
                            formatted_messages += f"{content} [/INST] "
                        elif role == 'function':
                            formatted_messages += f"Output: {content} [/INST] "
                        elif role == 'assistant':
                            formatted_messages += f"{content} </s><s>[INST] "

                    # Remove the trailing '<s>[INST] ' from the final output
                    if formatted_messages.endswith("<s>[INST] "):
                        formatted_messages = formatted_messages[:-10]

                return formatted_messages

            prompt = messages_to_prompt(messages)
            if messages[-1]["role"] != "function":
                prompt += \
                    "Let's explore this. By the way, " \
                    "I can run code on your machine " \
                    "by writing the code in a " \
                    "markdown code block. " \
                    "This works for shell, javascript, python, " \
                    "and applescript. " \
                    "I'm going to try to do this for your task. Anyway, "
            elif messages[-1]["role"] == "function" \
                    and messages[-1]["content"] != "No output":
                prompt += "Given the output of the code I just ran, "
            elif messages[-1]["role"] == "function" \
                    and messages[-1]["content"] == "No output":
                prompt += "Given the fact that the code I " \
                    "just ran produced no output, "

            if self.debug_mode:
                import builtins
                builtins.print("TEXT PROMPT SEND TO LLM:\n", prompt)

            # Run Code-Llama
            response = self.llama_instance(
                prompt,
                stream=True,
                temperature=self.temperature,
                stop=["</s>"],
                max_tokens=750
            )

        # Initialize message, function call trackers, and active block
        self.messages.append({})
        in_function_call = False
        llama_function_call_finished = False
        self.active_block = None

        for chunk in response:
            if self.use_azure \
                    and (
                        'choices' not in chunk
                        or len(chunk['choices']) == 0):
                # Azure OpenAI Service may return empty chunk
                continue

            if self.local:
                if "content" not in messages[-1]:
                    print("Invalid messages object:", self.messages)
                    chunk["choices"][0]["text"] = \
                        chunk["choices"][0]["text"].capitalize()
                    messages[-1]["role"] = "assistant"
                delta = {"content": chunk["choices"][0]["text"]}
            else:
                delta = chunk["choices"][0]["delta"]

            # Accumulate deltas into the last message in messages
            self.messages[-1] = merge_deltas(self.messages[-1], delta)

            # Check if we're in a function call
            if not self.local:
                condition = "function_call" in self.messages[-1]
            elif self.local:
                if "content" in self.messages[-1]:
                    condition = self.messages[-1][
                        "content"].count(
                            "```") % 2 == 1
                else:
                    # If it hasn't made "content" yet, we're certainly not
                    # in a function call.
                    condition = False

                if condition:
                    # We are in a function call.

                    # Check if we just entered a function call
                    if in_function_call is False:
                        # If so, end the last block,
                        self.end_active_block()

                        # Print newline if it was just a code block or
                        # user message (this just looks nice)
                        last_role = self.messages[-2]["role"]
                        if last_role == "user" or last_role == "function":
                            print()

                        # then create a new code block
                        self.active_block = CodeBlock()

                    # Remember we're in a function_call
                    in_function_call = True

                    # Now let's parse the function's arguments:
                    if not self.local:
                        # gpt-4
                        # Parse arguments and save to parsed_arguments,
                        # under function_call
                        if "arguments" in self.messages[-1]["function_call"]:
                            arguments = \
                                self.messages[-1]["function_call"]["arguments"]
                            new_parsed_arguments = parse_partial_json(
                                arguments)
                            if new_parsed_arguments:
                                # Only overwrite what we have if it's not None
                                # (which means it failed to parse)
                                self.messages[-1]["function_call"][
                                    "parsed_arguments"] = new_parsed_arguments

                    elif self.local:
                        # Code-Llama: Parse Current Code Block
                        if "content" in self.messages[-1]:
                            content = self.messages[-1]["content"]
                            if "```" in content:
                                # Split by "```" to get the last
                                # open code block
                                blocks = content.split("```")
                                current_code_block = blocks[-1]
                                lines = current_code_block.split("\n")
                                # Hasn't outputted a language yet
                                if content.strip() == "```":
                                    language = None
                                else:
                                    if lines[0] != "":
                                        language = lines[0].strip()
                                    else:
                                        language = "python"
                                        # In anticipation of its dumbassery
                                        # let's check if "pip"
                                        # is in there
                                        if len(lines) > 1:
                                            if lines[1].startswith("pip"):
                                                language = "shell"
                                # Join all lines except for the language line
                                code = '\n'.join(lines[1:]).strip("` \n")
                                arguments = {"code": code}
                                if language:
                                    if language == "bash":
                                        language = "shell"
                                    arguments["language"] = language

                            if "function_call" not in self.messages[-1]:
                                self.messages[-1]["function_call"] = {}

                            self.messages[-1][
                                "function_call"][
                                    "parsed_arguments"] \
                                = arguments

                else:
                    if in_function_call is True:
                        if self.local:
                            llama_function_call_finished = True
                    in_function_call = False
                    if self.active_block is None:
                        self.active_block = MessageBlock()
                self.active_block.update_from_message(self.messages[-1])
                # Check if we're finished
                if chunk["choices"][0]["finish_reason"] \
                        or llama_function_call_finished:
                    if chunk["choices"][
                        0]["finish_reason"] == "function_call" or \
                            llama_function_call_finished:

                        if self.debug_mode:
                            print("Running function:")
                            print(self.messages[-1])
                            print("---")

                        # Ask for user confirmation to run code
                        if self.auto_run is False:
                            self.active_block.end()
                            language = self.active_block.language
                            code = self.active_block.code

                            # Prompt user
                            response = input(
                                "  Would you like to run this code? "
                                "(y/n)\n\n  ")
                            print("")  # <- Aesthetic choice

                            if response.strip().lower() == "y":
                                # Create a new, identical block where
                                # the code will actually be run
                                self.active_block = CodeBlock()
                                self.active_block.language = language
                                self.active_block.code = code

                        else:
                            # User declined to run code.
                            self.active_block.end()
                            self.messages.append({
                                "role": "function",
                                "name": "run_code",
                                "content": "User decided not to run this code."
                            })
                            return

                        # Couldn't parse, no retry
                        if not self.local \
                            and "parsed_arguments" \
                                not in self.messages[-1]["function_call"]:

                            # After collecting some data via
                            # the below instruction to users,
                            # This is the most common failure pattern:
                            # https://github.com/KillianLucas/open-interpreter/issues/41

                            self.messages.append({
                                "role": "function",
                                "name": "run_code",
                                "content": """Your function call "
                                "could not be parsed. "
                                "Please use ONLY the `run_code` function, "
                                "which takes two parameters: "
                                "`code` and `language`. "
                                "Your response should be formatted "
                                "as a JSON."""
                            })

                            self.respond()
                            return

                        # Create/retrieve Code Interpreter for lang
                        language = self.messages[-1][
                            "function_call"][
                                "parsed_arguments"][
                                    "language"]
                        if language not in self.code_interpreters:
                            self.code_interpreters[language] = \
                                CodeInterpreter(language, self.debug_mode)
                        code_interpreter = self.code_interpreters[language]

                        # Let this Code Interpreter control the active_block
                        code_interpreter.active_block = self.active_block
                        code_interpreter.run()

                        # End the active_block
                        self.active_block.end()

                        # Append Output to Messages
                        # No Output ("") may cause hallucination
                        self.messages.append({
                            "role": "function",
                            "name": "run_code",
                            "content": self.active_block.output
                            if self.active_block.output else "No output"
                            })

                        # Go around again
                        self.respond()

                    if chunk["choices"][0]["finish_reason"] != "function_call":
                        # Done!

                        # Code Llama outputs "###" at message end
                        if self.local and "content" in self.messages[-1]:
                            self.messages[-1]["content"] = self.messages[-1][
                                "content"].strip().rstrip("#")
                            self.active_block.update_from_message(
                                self.messages[-1])
                            time.sleep(0.1)
                        self.active_block.end()
                        return

    def _print_welcome_message(self):
        current_version = pkg_resources.get_distribution(
            "open-interpreter").version
        print(
            f"\n Hello, Welcome to [bold]● Open Interpreter[/bold]."
            f"(v{current_version})\n")
