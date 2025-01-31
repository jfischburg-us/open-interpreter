"""
interpreter.py - Main module for Open Interpreter chatbot.

This module contains the Interpreter class which manages conversations
with users. It handles sending messages to AI models, running code,
displaying output, and more.

Key components:
- Interpreter: Main class for chat sessions.
- cli: Handles command line interface.
- utils: Helper functions.
- message_block: Display conversational messages.
- code_block: Display and run code.
- code_interpreter: Execute code in different languages.

The Interpreter can be used directly for conversations or via the
command line interface. By default it will use GPT-3.5 Turbo from
Anthropic for dialog.

Usage:
    >>> from interpreter import Interpreter
    >>> interpreter = Interpreter()
    >>> interpreter.chat()

"""
import os
import time
import traceback
import json
import platform
import getpass
import builtins
import readline

import inquirer
import litellm
import requests
import tokentrim as tt
from rich import print
from rich.markdown import Markdown
from rich.rule import Rule
from .code_interpreter import CodeInterpreter

from .cli import cli
from .utils import merge_deltas, parse_partial_json
from .message_block import MessageBlock
from .code_block import CodeBlock
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
                "description": "The programming language",
                "enum": ["python", "R", "shell", "applescript", "javascript", "html"],
            },
            "code": {"type": "string", "description": "The code to execute"},
        },
        "required": ["language", "code"],
    },
}

# Message for when users don't have an OpenAI API key.
MISSING_API_KEY_MESSAGE = """> OpenAI API key not found

To use `GPT-4` (recommended) please provide an OpenAI API key.

To use `Code-Llama` (free but less capable) press `enter`.
"""

# Message for when users don't have an OpenAI API key.
MISSING_AZURE_INFO_MESSAGE = """> Azure OpenAI Service API info not found

To use `GPT-4` (recommended) please provide an Azure OpenAI API key, a API base, a deployment name and a API version.

To use `Code-Llama` (free but less capable) press `enter`.
"""

# Message to confirm usage mode.
CONFIRM_MODE_MESSAGE = """
**Open Interpreter** will require approval before running code. Use `interpreter -y` to bypass this.

Press `CTRL-C` to exit.
"""


class Interpreter:
    """
    A class for interpreting and executing scripts.

    This class provides methods to interpret and execute scripts written in a
    custom scripting language. It can handle various commands and perform
    actions based on the script content.

    Attributes:
        script (str): The script to be interpreted and executed.
        environment (dict): The environment in which the script is executed.

    Methods:
        execute(): Execute the provided script in the specified environment.
        add_variable(name, value): Add a variable to the environment.
        remove_variable(name): Remove a variable from the environment.

    Example usage:
        interpreter = Interpreter(script="print('Hello, world!')")
        interpreter.add_variable("name", "John")
        interpreter.execute()
    """

    def __init__(self):
        """
        Initializes an instance of the Interpreter class.
        """
        self.messages = []
        self.temperature = 0.001
        self.api_key = None
        self.auto_run = False
        self.local = False
        self.model = "gpt-4"
        self.debug_mode = False
        self.api_base = None  # Will set it to whatever OpenAI wants
        self.context_window = 2000  # For local models only
        self.max_tokens = 750  # For local models only
        # Azure OpenAI
        self.use_azure = False
        self.azure_api_base = None
        self.azure_api_version = None
        self.azure_deployment_name = None
        self.azure_api_type = "azure"

        # Get default system message
        here = os.path.abspath(os.path.dirname(__file__))
        with open(
            os.path.join(here, "system_message.txt"), "r", encoding="utf8"
        ) as default_message:
            self.system_message = default_message.read().strip()

        # Store Code Interpreter instances for each language
        self.code_interpreters = {}

        # No active block to start
        # (blocks are visual representation of messages on the terminal)
        self.active_block = None

        # Note: While Open Interpreter can use Llama, we will prioritize gpt-4.
        # gpt-4 is faster, smarter, can call functions, and is all-around easier to use.
        # This makes gpt-4 better aligned with Open Interpreters priority to be easy to use.
        self.llama_instance = None

    def cli(self):
        """
        Modifies the current instance of Interpreter according to command line flags,
        then runs chat.
        """
        # The cli takes the current instance of Interpreter,
        # modifies it according to command line flags, then runs chat.
        cli(self)

    def get_info_for_system_message(self):
        """
        Retrieves relevant information for the system message.
        """

        info = ""

        # Add user info
        username = getpass.getuser()
        current_working_directory = os.getcwd()
        operating_system = platform.system()

        info += (
            f"[User Info]\nName: {username}\n"
            f"CWD: {current_working_directory}\n"
            f"OS: {operating_system}"
        )

        if not self.local:
            # Open Procedures is an open-source database of tiny,
            # structured coding tutorials.
            # We can query it semantically and append relevant
            # tutorials/procedures to our system message:

            # Use the last two messages' content or function
            # call to semantically search
            query = []
            for message in self.messages[-2:]:
                message_for_semantic_search = {"role": message["role"]}
                if "content" in message:
                    message_for_semantic_search["content"] = message["content"]
                if (
                    "function_call" in message
                    and "parsed_arguments" in message["function_call"]
                ):
                    message_for_semantic_search["function_call"] = message[
                        "function_call"
                    ]["parsed_arguments"]
                query.append(message_for_semantic_search)

            # Use them to query Open Procedures
            url = "https://open-procedures.replit.app/search/"

            try:
                relevant_procedures = requests.get(
                    url, data=json.dumps(query), timeout=15
                ).json()["procedures"]
                info += (
                    "\n\n# Recommended Procedures\n"
                    + "\n---\n".join(relevant_procedures)
                    + "\nIn your plan, include steps and, if present, "
                    + "**EXACT CODE SNIPPETS** (especially for deprecation notices, "
                    + "**WRITE THEM INTO YOUR PLAN -- "
                    + "underneath each numbered step** as they will "
                    + "VANISH once you execute your first line of code, "
                    + "so WRITE THEM DOWN NOW if you need them) from the above "
                    + "procedures if they are relevant to the task. "
                    + "Again, include **VERBATIM CODE SNIPPETS** from the "
                    + "procedures above if they are relevent to the task "
                    + "**directly in your plan.**"
                )
            except:
                # For someone, this failed for a super secure
                # SSL reason. Since it's not stricly necessary,
                # let's worry about that another day.
                # Should probably log this somehow though.
                pass

        elif self.local:
            # Tell Code-Llama how to run code.
            info += (
                "\n\nTo run code, write a fenced code block "
                "(i.e ```python, R or ```shell) in markdown. "
                "When you close it with ```, it will be run. "
                "You'll then be given its output."
            )
            # We make references in system_message.txt to the "function" it can call, "run_code".

        return info

    def reset(self):
        """
        Resets the interpreter by clearing messages and code interpreters.
        """
        self.messages = []
        self.code_interpreters = {}

    def load(self, messages):
        """
        Loads a list of messages into the interpreter.

        Args:
            messages (list): The list of messages to load.
        """
        self.messages = messages

    def handle_undo(self, arguments):
        """
        Removes all messages after the most recent user entry (and the entry itself).

        Args:
            arguments: Arguments for undo operation. Not used in this method.
        """
        # Removes all messages after the most recent user entry (and the entry itself).
        # Therefore user can jump back to the latest point of conversation.
        # Also gives a visual representation of the messages removed.

        if len(self.messages) == 0:
            return
        # Find the index of the last 'role': 'user' entry
        last_user_index = None
        for i, message in enumerate(self.messages):
            if message.get("role") == "user":
                last_user_index = i

        removed_messages = []

        # Remove all messages after the last 'role': 'user'
        if last_user_index is not None:
            removed_messages = self.messages[last_user_index:]
            self.messages = self.messages[:last_user_index]

        print("")  # Aesthetics.

        # Print out a preview of what messages were removed.
        for message in removed_messages:
            if "content" in message and message["content"] is not None:
                print(
                    Markdown(f"**Removed message:** `\"{message['content'][:30]}...\"`")
                )
            elif "function_call" in message:
                print(Markdown("**Removed codeblock**"))

        print("")  # Aesthetics.

    def handle_help(self, arguments):
        """
        Displays help information for available commands.

        Args:
            arguments: Arguments for help operation. Not used in this method.
        """
        commands_description = {
            "%debug [true/false]": "Toggle debug mode. "
            "Without arguments or with 'true', it enters debug mode. "
            "With 'false', it exits debug mode.",
            "%reset": "Resets the current session.",
            "%undo": "Remove previous messages and its response from the message history.",
            "%save_message [path]": "Saves messages to a specified JSON path. "
            "If no path is provided, it defaults to 'messages.json'.",
            "%load_message [path]": "Loads messages from a specified JSON path. "
            "If no path is provided, it defaults to 'messages.json'.",
            "%help": "Show this help message.",
        }

        base_message = ["> **Available Commands:**\n\n"]

        # Add each command and its description to the message
        for cmd, desc in commands_description.items():
            base_message.append(f"- `{cmd}`: {desc}\n")

        additional_info = [
            (
                "\n\nFor further assistance, please join our community "
                "Discord or consider contributing to the "
                "project's development."
            )
        ]

        # Combine the base message with the additional info
        full_message = base_message + additional_info

        print(Markdown("".join(full_message)))

    def handle_debug(self, arguments=None):
        """
        Toggles debug mode based on provided argument.

        Args:
            arguments (str): Argument to toggle debug mode. Can be "true" or "false".
        """
        if arguments == "" or arguments == "true":
            print(Markdown("> Entered debug mode"))
            print(self.messages)
            self.debug_mode = True
        elif arguments == "false":
            print(Markdown("> Exited debug mode"))
            self.debug_mode = False
        else:
            print(Markdown("> Unknown argument to debug command."))

    def handle_reset(self, arguments):
        """
        Resets the interpreter session.

        Args:
            arguments: Arguments for reset operation. Not used in this method.
        """
        self.reset()
        print(Markdown("> Reset Done"))

    def default_handle(self, arguments):
        """
        Default handler for unknown commands.
        Displays an unknown command message and shows help information.

        Args:
            arguments: Arguments for default operation. Not used in this method.
        """
        print(Markdown("> Unknown command"))
        self.handle_help(arguments)

    def handle_save_message(self, json_path):
        """
        Saves messages to a specified JSON path.

        Args:
            json_path (str): The path where to save the messages JSON file.
        """
        if json_path == "":
            json_path = "messages.json"
        if not json_path.endswith(".json"):
            json_path += ".json"
        with open(json_path, "w", encoding="utf8") as file_save:
            json.dump(self.messages, file_save, indent=2)

        print(Markdown(f"> messages json export to {os.path.abspath(json_path)}"))

    def handle_load_message(self, json_path):
        """
        Loads messages from a specified JSON path.

        Args:
            json_path (str): The path from where to load the messages JSON file.
        """
        if json_path == "":
            json_path = "messages.json"
        if not json_path.endswith(".json"):
            json_path += ".json"
        with open(json_path, "r", encoding="utf8") as load_message:
            self.load(json.load(load_message))

        print(Markdown(f"> messages json loaded from {os.path.abspath(json_path)}"))

    def handle_command(self, user_input):
        """
        Handles a user command.

        Args:
            user_input (str): The user's input command string.
        """
        # split the command into the command and the arguments, by the first whitespace
        switch = {
            "help": self.handle_help,
            "debug": self.handle_debug,
            "reset": self.handle_reset,
            "save_message": self.handle_save_message,
            "load_message": self.handle_load_message,
            "undo": self.handle_undo,
        }

        user_input = user_input[1:].strip()  # Capture the part after the `%`
        command = user_input.split(" ")[0]
        arguments = user_input[len(command) :].strip()
        action = switch.get(
            command, self.default_handle
        )  # Get the function from the dictionary, or default_handle if not found
        action(arguments)  # Execute the function

    def chat(self, message=None, return_messages=False):
        """
        Starts an interactive chat or responds to a single message.

        Args:
            message (str): A single message to respond to.
            If None, starts an interactive chat instead. Defaults to None.
            return_messages (bool): Whether or not to return the list of
            messages after chatting. Defaults to False.

        Returns:
            list: If return_messages is True, returns a list of all
            messages from the chat session.
        """
        # Connect to an LLM (an large language model)
        if not self.local:
            # gpt-4
            self.verify_api_key()

        # ^ verify_api_key may set self.local to True,
        # so we run this as an 'if', not 'elif':
        if self.local:
            # Code-Llama
            if self.llama_instance is None:
                # Find or install Code-Llama
                try:
                    self.llama_instance = get_hf_llm(
                        self.model, self.debug_mode, self.context_window
                    )
                    if self.llama_instance is None:
                        # They cancelled.
                        return
                except:
                    traceback.print_exc()

            # Check if llama_instance was properly initialized
            if not callable(self.llama_instance):
                raise ValueError("Failed to initialize llama instance")

            # If it didn't work, apologize and switch to GPT-4

            print(
                Markdown(
                    "".join(
                        [
                            f"> Failed to install `{self.model}`.\n\n",
                            "**Common Fixes:** You can follow our simple setup docs "
                            f"at the link below to resolve common errors.\n\n```\n"
                            f"https://github.com/KillianLucas"
                            f"/open-interpreter/tree/main/docs\n```",
                            f"\n\n**If you've tried that and you're still "
                            f"getting an error, "
                            f"we have likely not built the proper `{self.model}` "
                            f"support for your system.**",
                            "\n\n*( Running language models locally "
                            f"is a difficult task!*"
                            f"If you have insight into the best way "
                            f"to implement this across "
                            f"platforms/architectures, please join the "
                            f"Open Interpreter community "
                            f"Discord and consider contributing the "
                            f"project's development. )",
                            "\n\nPress enter to switch to `GPT-4` (recommended).",
                        ]
                    )
                )
            )
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
            if self.use_azure:
                notice_model = f"{self.azure_deployment_name} (Azure)"
            else:
                notice_model = f"{self.model.upper()}"
            welcome_message += (
                f"\n> Model set to `{notice_model}`\n\n"
                f"**Tip:** To run locally, use `interpreter --local`"
            )

        if self.local:
            welcome_message += f"\n> Model set to `{self.model}`"

        # If not auto_run, tell the user we'll ask permission to run code
        # We also tell them here how to exit Open Interpreter
        if not self.auto_run:
            welcome_message += "\n\n" + CONFIRM_MODE_MESSAGE

        welcome_message = welcome_message.strip()

        # Print welcome message with newlines on either side (aesthetic choice)
        # unless we're starting with a blockquote (aesthetic choice)
        if welcome_message != "":
            if welcome_message.startswith(">"):
                print(Markdown(welcome_message), "")
            else:
                print("", Markdown(welcome_message), "")

        # Check if `message` was passed in by user
        if message:
            # If it was, we respond non-interactivley
            self.messages.append({'content': ''})
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

                # Use `readline` to let users up-arrow to previous user messages,
                # which is a common behavior in terminals.
                readline.add_history(user_input)

                # If the user input starts with a `%` or `/`, it's a command
                if user_input.startswith("%") or user_input.startswith("/"):
                    self.handle_command(user_input)
                    continue

                # Add the user message to self.messages
                self.messages.append({'content': ''})
                self.messages.append({"role": "user", "content": user_input})

                # Respond, but gracefully handle CTRL-C / KeyboardInterrupt
                try:
                    self.respond()
                except KeyboardInterrupt:
                    pass
                finally:
                    # Always end the active block. Multiple Live displays = issues
                    self.end_active_block()

        if return_messages:
            return self.messages

    def verify_api_key(self):
        """
        Makes sure we have an AZURE_API_KEY or OPENAI_API_KEY.
        """

        # Initialize response
        response = None

        if self.use_azure:
            all_env_available = (
                ("AZURE_API_KEY" in os.environ or "OPENAI_API_KEY" in os.environ)
                and "AZURE_API_BASE" in os.environ
                and "AZURE_API_VERSION" in os.environ
                and "AZURE_DEPLOYMENT_NAME" in os.environ
            )
            if all_env_available:
                self.api_key = (
                    os.environ.get("AZURE_API_KEY") or os.environ["OPENAI_API_KEY"]
                )
                self.azure_api_base = os.environ["AZURE_API_BASE"]
                self.azure_api_version = os.environ["AZURE_API_VERSION"]
                self.azure_deployment_name = os.environ["AZURE_DEPLOYMENT_NAME"]
                self.azure_api_type = os.environ.get("AZURE_API_TYPE", "azure")
            else:
                # This is probably their first time here!
                self._print_welcome_message()
                time.sleep(1)

                print(Rule(style="white"))

                print(Markdown(MISSING_AZURE_INFO_MESSAGE), "", Rule(style="white"), "")
                response = input("Azure OpenAI API key: ")

                if response == "":
                    # User pressed `enter`, requesting Code-Llama

                    print(
                        Markdown(
                            "> Switching to `Code-Llama`...\n\n"
                            "**Tip:** Run `interpreter --local` to "
                            "automatically use `Code-Llama`."
                        ),
                        "",
                    )
                    time.sleep(2)
                    print(Rule(style="white"))

                    # Temporarily, for backwards (behavioral) compatability,
                    # we've moved this part of llama_2.py here.
                    # AND BELOW.
                    # This way, when folks hit interpreter --local,
                    # they get the same experience as before.

                    print(
                        "",
                        Markdown(
                            "**Open Interpreter** will use `Code Llama` for local execution. "
                            "Use your arrow keys to set up the model."
                        ),
                        "",
                    )

                    models = {
                        "7B": "TheBloke/CodeLlama-7B-Instruct-GGUF",
                        "13B": "TheBloke/CodeLlama-13B-Instruct-GGUF",
                        "34B": "TheBloke/CodeLlama-34B-Instruct-GGUF",
                    }

                    parameter_choices = list(models.keys())
                    questions = [
                        inquirer.List(
                            "param",
                            message="Parameter count (smaller is faster, larger is more capable)",
                            choices=parameter_choices,
                        )
                    ]
                    answers = inquirer.prompt(questions)
                    if answers is not None:
                        chosen_param = answers["param"]

                        # THIS is more in line with the future.
                        # You just say the model you want by name:
                        self.model = models[chosen_param]
                        self.local = True
                    else:
                        print("No answer provided. Please try again.")
                    return

                else:
                    self.api_key = response
                    self.azure_api_base = input("Azure OpenAI API base: ")
                    self.azure_deployment_name = input(
                        "Azure OpenAI deployment name of GPT: "
                    )
                    self.azure_api_version = input("Azure OpenAI API version: ")
                    print(
                        "",
                        Markdown(
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
                            "on Windows."
                        ),
                        "",
                    )
                    time.sleep(2)
                    print(Rule(style="white"))

            litellm.openai.api_type = self.azure_api_type
            litellm.api_base = self.azure_api_base
            litellm.api_version = self.azure_api_version
            litellm.api_key = self.api_key
        else:
            if self.api_key is None:
                if "OPENAI_API_KEY" in os.environ:
                    self.api_key = os.environ["OPENAI_API_KEY"]
                else:
                    # This is probably their first time here!
                    self._print_welcome_message()
                    time.sleep(1)

                    print(Rule(style="white"))

                    print(
                        Markdown(MISSING_API_KEY_MESSAGE), "", Rule(style="white"), ""
                    )
                    response = input("OpenAI API key: ")

                    if response == "":
                        # User pressed `enter`, requesting Code-Llama

                        print(
                            Markdown(
                                "> Switching to `Code-Llama`...\n\n**"
                                "Tip:** Run `interpreter --local` "
                                "to automatically use `Code-Llama`."
                            ),
                            "",
                        )
                        time.sleep(2)
                        print(Rule(style="white"))

                        # Temporarily, for backwards (behavioral) compatability,
                        # we've moved this part of llama_2.py here.
                        # AND ABOVE.
                        # This way, when folks hit interpreter --local,
                        # they get the same experience as before.

                        print(
                            "",
                            Markdown(
                                "**Open Interpreter** will use `Code Llama` "
                                "for local execution. "
                                "Use your arrow keys to set up the model."
                            ),
                            "",
                        )

                        models = {
                            "7B": "TheBloke/CodeLlama-7B-Instruct-GGUF",
                            "13B": "TheBloke/CodeLlama-13B-Instruct-GGUF",
                            "34B": "TheBloke/CodeLlama-34B-Instruct-GGUF",
                        }

                        parameter_choices = list(models.keys())
                        questions = [
                            inquirer.List(
                                "param",
                                message=(
                                    "Parameter count: smaller is faster, "
                                    "larger is more capable)"
                                ),
                                choices=parameter_choices,
                            )
                        ]
                        answers = inquirer.prompt(questions)
                        if answers is not None:
                            chosen_param = answers["param"]
                        else:
                            chosen_param = "7B"  # Or any other default value

                        # THIS is more in line with the future.
                        # You just say the model you want by name:
                        self.model = models[chosen_param]
                        self.local = True

                        return

                    else:
                        self.api_key = response
                        print(
                            "",
                            Markdown(
                                "**Tip:** To save this key for later, "
                                "run `export OPENAI_API_KEY=your_api_key` "
                                "on Mac/Linux or `setx OPENAI_API_KEY your_api_key` "
                                "on Windows."
                            ),
                            "",
                        )
                        time.sleep(2)
                        print(Rule(style="white"))

            litellm.api_key = self.api_key
            if self.api_base:
                litellm.api_base = self.api_base

    def end_active_block(self):
        """
        Ends the currently active block.
        """
        if self.active_block:
            self.active_block.end()
            self.active_block = None

    def respond(self):
        """
        Responds to the most recent message in the messages list.
        """

        # Initialize response
        response = None

        # Add relevant info to system_message
        # (e.g. current working directory, username, os, etc.)
        info = self.get_info_for_system_message()

        # This is hacky, as we should have a different (minified) prompt for CodeLLama,
        # but for now, to make the prompt shorter and remove "run_code" references,
        # just get the first 2 lines:
        if self.local:
            self.system_message = "\n".join(self.system_message.split("\n")[:2])
            self.system_message += (
                "\nOnly do what the user asks you to do, "
                "then ask what they'd like to do next."
            )

        system_message = self.system_message + "\n\n" + info

        if self.local:
            messages = tt.trim(
                self.messages,
                max_tokens=(self.context_window - self.max_tokens - 25),
                system_message=system_message,
            )
        else:
            messages = tt.trim(self.messages, self.model, system_message=system_message)

        if self.debug_mode:
            print("\n", "Sending `messages` to LLM:", "\n")
            print(messages)
            print()

        # Make LLM call
        if not self.local:
            # GPT

            error = ""

            self.messages.append({})  # Add empty message dict
            self.messages.append({'content': ''})

            # New code to check and add 'content'
            if "content" not in self.messages[-1]:
                self.messages[-1]["content"] = ""

            for _ in range(3):  # 3 retries
                try:
                    if self.use_azure:
                        if isinstance(messages, tuple):
                            messages = messages[0]
                            response = litellm.completion(
                                f"azure/{self.azure_deployment_name}",
                                messages=messages,
                                functions=[function_schema],
                                temperature=self.temperature,
                                stream=True,
                            )
                    else:
                        if self.api_base:
                            if isinstance(messages, tuple):
                                messages = messages[0]
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
                            if isinstance(messages, tuple):
                                messages = messages[0]
                                # Normal OpenAI call
                                response = litellm.completion(
                                    model=self.model,
                                    messages=messages,
                                    functions=[function_schema],
                                    stream=True,
                                    temperature=self.temperature,
                                )
                    break
                except:
                    if self.debug_mode:
                        traceback.print_exc()
                    error = traceback.format_exc()
                    time.sleep(3)
            else:
                raise Exception(error)

        elif self.local:
            # Code-Llama

            # Convert messages to prompt
            # (This only works if the first message is the only system message)

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
                            f"{message['role'].capitalize()}: {message['content']}\n"
                        )
                    formatted_messages = formatted_messages.strip()

                else:
                    # Llama prompt template

                    # Extracting the system prompt and initializing the formatted string with it.
                    system_prompt = messages[0]["content"]
                    formatted_messages = (
                        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
                    )

                    # Loop starting from the first user message
                    for index, item in enumerate(messages[1:]):
                        role = item["role"]
                        content = item["content"]

                        if role == "user":
                            formatted_messages += f"{content} [/INST] "
                        elif role == "function":
                            formatted_messages += f"Output: {content} [/INST] "
                        elif role == "assistant":
                            formatted_messages += f"{content} </s><s>[INST] "

                    # Remove the trailing '<s>[INST] ' from the final output
                    if formatted_messages.endswith("<s>[INST] "):
                        formatted_messages = formatted_messages[:-10]

                return formatted_messages

            prompt = messages_to_prompt(messages)
            # Lmao i can't believe this works (it does need this btw)
            if isinstance(messages[-1], dict):
                if messages[-1]["role"] != "function":
                    prompt += (
                        "Let's explore this. "
                        "By the way, I can run code on your machine by writing the code "
                        "in a markdown code block. "
                        "This works for shell, javascript, python, R, and applescript. "
                        "I'm going to try to do this for your task. Anyway, "
                    )
                elif (
                    messages[-1]["role"] == "function"
                    and messages[-1]["content"] != "No output"
                ):
                    prompt += "Given the output of the code I just ran, "
                elif (
                    messages[-1]["role"] == "function"
                    and messages[-1]["content"] == "No output"
                ):
                    prompt += (
                        "Given the fact that the code I just ran produced no output, "
                    )

            if self.debug_mode:
                # we have to use builtins bizarrely! because rich.print interprets
                # "[INST]" as something meaningful

                builtins.print("TEXT PROMPT SEND TO LLM:\n", prompt)

            # Run Code-Llama
            # Initialize response before assignment
            response = None
            if self.llama_instance is not None:
                response = self.llama_instance(
                    prompt,
                    stream=True,
                    temperature=self.temperature,
                    stop=["</s>"],
                    # context window is set to 1800
                    # messages are trimmed to 1000
                    # 700 seems nice
                    max_tokens=750,
                )
            else:
                raise ValueError("Failed to initialize llama instance")

        # Initialize message, function call trackers, and active block
        self.messages.append({'content': ''})
        in_function_call = False
        llama_function_call_finished = False
        self.active_block = None

        if response is None or not hasattr(response, "__iter__"):
            raise ValueError("Response is either None or not iterable")
        else:
            for chunk in response:
                if self.use_azure and (
                    "choices" not in chunk or len(chunk["choices"]) == 0
                ):
                    # Azure OpenAI Service may return empty chunk
                    continue

                # Initialize condition with a default value
                condition = False

                if self.local:
                    # Check if messages[-1] is a dictionary
                    if isinstance(messages[-1], dict):
                        if "content" not in messages[-1]:
                            # This is the first chunk. We'll need to capitalize it,
                            # because our prompt ends in a ", "
                            chunk["choices"][0]["text"] = chunk["choices"][0][
                                "text"
                            ].capitalize()
                            # We'll also need to add "role: assistant",
                            # CodeLlama will not generate this
                            messages[-1]["role"] = "assistant"
                        delta = {"content": chunk["choices"][0]["text"]}
                    else:
                        delta = chunk["choices"][0]["delta"]
                else:
                    delta = chunk["choices"][0]["delta"]

                # Accumulate deltas into the last message in messages
                self.messages[-1] = merge_deltas(self.messages[-1], delta)

                # Check if 'content' exists in the updated message
                if "content" not in self.messages[-1]:
                    # If not, initialize it with an empty string
                    self.messages[-1]["content"] = ""

                # Check if we're in a function call
                if not self.local:
                    condition = "function_call" in self.messages[-1]
                elif self.local:
                    # Since Code-Llama can't call functions,
                    # we just check if we're in a code block.
                    # This simply returns true if the number of
                    # "```" in the message is odd.
                    if "content" in self.messages[-1]:
                        condition = self.messages[-1]["content"].count("```") % 2 == 1
                    elif self.local:
                        # If it hasn't made "content" yet,
                        # we're certainly not in a function call.
                        condition = False

                if condition:
                    # We are in a function call.

                    # Check if we just entered a function call
                    if in_function_call is False:
                        # If so, end the last block,
                        self.end_active_block()

                        # Print newline if it was just a code block or user message
                        # (this just looks nice)
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
                        # Parse arguments and save to parsed_arguments, under function_call
                        if "arguments" in self.messages[-1]["function_call"]:
                            arguments = self.messages[-1]["function_call"]["arguments"]
                            new_parsed_arguments = parse_partial_json(arguments)
                            if new_parsed_arguments:
                                # Only overwrite what we have if it's not None
                                # (which means it failed to parse)
                                self.messages[-1]["function_call"][
                                    "parsed_arguments"
                                ] = new_parsed_arguments

                    elif self.local:
                        # Code-Llama
                        # Parse current code block and save to parsed_arguments,
                        # under function_call

                        # Initialize arguments with a default value
                        arguments = {}

                        if "content" in self.messages[-1]:
                            content = self.messages[-1]["content"]

                            if "```" in content:
                                # Split by "```" to get the last open code block
                                blocks = content.split("```")

                                current_code_block = blocks[-1]

                                lines = current_code_block.split("\n")

                                if (
                                    content.strip() == "```"
                                ):  # Hasn't outputted a language yet
                                    language = None
                                else:
                                    if lines[0] != "":
                                        language = lines[0].strip()
                                    else:
                                        language = "python"
                                        # In anticipation of its dumbassery let's check
                                        # if "pip" is in there
                                        if len(lines) > 1:
                                            if lines[1].startswith("pip"):
                                                language = "shell"

                                # Join all lines except for the language line
                                code = "\n".join(lines[1:]).strip("` \n")

                                arguments = {"code": code}
                                if (
                                    language
                                ):  # We only add this if we have it-- the second we have it,
                                    # an interpreter gets fired up (I think? maybe I'm wrong)
                                    if language == "bash":
                                        language = "shell"
                                    arguments["language"] = language

                            # Code-Llama won't make a "function_call" property
                            # for us to store this under, so:
                            if "function_call" not in self.messages[-1]:
                                self.messages[-1]["function_call"] = {}

                            self.messages[-1]["function_call"][
                                "parsed_arguments"
                            ] = arguments

                else:
                    # We are not in a function call.

                    # Check if we just left a function call
                    if in_function_call is True:
                        if self.local:
                            # This is the same as when gpt-4 gives finish_reason as function_call.
                            # We have just finished a code block, so now we should run it.
                            llama_function_call_finished = True

                    # Remember we're not in a function_call
                    in_function_call = False

                    # If there's no active block,
                    if self.active_block is None:
                        # Create a message block
                        self.active_block = MessageBlock()

                # Update active_block
                if self.active_block is not None:
                    self.active_block.update_from_message(self.messages[-1])

                # Check if we're finished
                if chunk["choices"][0]["finish_reason"] or llama_function_call_finished:
                    if (
                        chunk["choices"][0]["finish_reason"] == "function_call"
                        or llama_function_call_finished
                    ):
                        # Time to call the function!
                        # (Because this is Open Interpreter, we only have one function.)

                        if self.debug_mode:
                            print("Running function:")
                            print(self.messages[-1])
                            print("---")

                        # Ask for user confirmation to run code
                        if self.auto_run is False:
                            if isinstance(self.active_block, CodeBlock):
                                code = self.active_block.code
                                self.active_block.end()
                                language = self.active_block.language
                                code = self.active_block.code

                            # Prompt user
                            response = input(
                                "  Would you like to run this code? (y/n)\n\n  "
                            )
                            print("")  # <- Aesthetic choice

                            if response.strip().lower() == "y":
                                # Create a new, identical block where the code will actually be run
                                if isinstance(self.active_block, CodeBlock):
                                    code = self.active_block.code
                                    language = self.active_block.language
                                    self.active_block.code = code

                            else:
                                # User declined to run code.
                                if self.active_block is not None:
                                    self.active_block.end()
                                self.messages.append(
                                    {
                                        "role": "function",
                                        "name": "run_code",
                                        "content": "User decided not to run this code.",
                                    }
                                )
                                return

                        # If we couldn't parse its arguments, we need to try again.
                        if (
                            not self.local
                            and "parsed_arguments"
                            not in self.messages[-1]["function_call"]
                        ):
                            # After collecting some data via the below instruction to users,
                            # This is the most common failure pattern:
                            # https://github.com/KillianLucas/open-interpreter/issues/41

                            # print("> Function call could not be parsed.\n\n
                            # Please open an issue on Github
                            # (openinterpreter.com, click Github) and paste the following:")
                            # print("\n", self.messages[-1]["function_call"], "\n")
                            # time.sleep(2)
                            # print("Informing the language model and continuing...")

                            # Since it can't really be fixed without something complex,
                            # let's just berate the LLM then go around again.

                            self.messages.append(
                                {
                                    "role": "function",
                                    "name": "run_code",
                                    "content": (
                                        """Your function call could not be parsed. "
                                        "Please use ONLY the `run_code` function, "
                                        "which takes two parameters: "
                                        "`code` and `language`. "
                                        "Your response should be formatted as a JSON.""",
                                    ),
                                }
                            )

                            self.respond()
                            return

                        # Create or retrieve a Code Interpreter for this language
                        language = self.messages[-1]["function_call"][
                            "parsed_arguments"
                        ]["language"]
                        if language not in self.code_interpreters:
                            self.code_interpreters[language] = CodeInterpreter(
                                language, self.debug_mode
                            )
                        code_interpreter = self.code_interpreters[language]

                        # Let this Code Interpreter control the active_block
                        code_interpreter.active_block = self.active_block
                        code_interpreter.run()

                        # End the active_block
                        if self.active_block is not None:
                            self.active_block.end()

                        # Append the output to messages
                        # Explicitly tell it if there was no output
                        # (sometimes "" = hallucinates output)
                        self.messages.append(
                            {
                                "role": "function",
                                "name": "run_code",
                                "content": self.active_block.output
                                if self.active_block and self.active_block.output
                                else "No output",
                            }
                        )

                        # Go around again
                        self.respond()

                    if chunk["choices"][0]["finish_reason"] != "function_call":
                        # Done!

                        # Code Llama likes to output "###"
                        # at the end of every message for some reason
                        if self.local and "content" in self.messages[-1]:
                            self.messages[-1]["content"] = (
                                self.messages[-1]["content"].strip().rstrip("#")
                            )
                            if self.active_block is not None:
                                self.active_block.update_from_message(self.messages[-1])
                                time.sleep(0.1)
                                self.active_block.end()
                        return

    def _print_welcome_message(self):
        """
        Prints a welcome message for the user.
        """
        print(
            "", Markdown("●"), "", Markdown("\nWelcome to **Open Interpreter**.\n"), ""
        )
