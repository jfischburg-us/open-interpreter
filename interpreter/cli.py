"""
Command-line interface (CLI) for interacting with an AI interpreter.

This module provides a command-line interface to interact with an AI
interpreter instance. It allows users to modify the interpreter's behavior
using various command-line flags and then run a chat session.

Args:
    interpreter: An instance of the AI interpreter to be configured
    and used for chat interactions.

Command-Line Flags:
    -y, --yes: Execute code without user confirmation.
    -f, --fast: Use the gpt-3.5-turbo model instead of gpt-4 for
    faster responses.
    -l, --local: Run in fully local mode with code-llama.
    -d, --debug: Print extra debugging information.
    --use-azure: Use Azure OpenAI Services instead of default settings.

Example:
    To run a chat session with the AI interpreter and enable the fast mode,
    you can use the following command:

    ```
    python your_script.py -f
    ```

Note:
    This module is designed to make it easy to configure and run chat
    interactions with the AI interpreter from the command line,
    allowing for quick adjustments to the interpreter's behavior
    based on user preferences.

Raises:
    None: This module does not raise any custom exceptions.

"""
import argparse


def cli(interpreter):
    """
    Takes an instance of interpreter.
    Modifies it according to command line flags, then runs chat.
    """

    # Setup CLI
    parser = argparse.ArgumentParser(description='Chat with Open Interpreter.')

    parser.add_argument('-y',
                        '--yes',
                        action='store_true',
                        help='execute code without user confirmation')
    parser.add_argument('-f',
                        '--fast',
                        action='store_true',
                        help='use gpt-3.5-turbo instead of gpt-4')
    parser.add_argument('-l',
                        '--local',
                        action='store_true',
                        help='run fully local with code-llama')
    parser.add_argument('-d',
                        '--debug',
                        action='store_true',
                        help='prints extra information')
    parser.add_argument(
      '--use-azure',
      action='store_true',
      help='use Azure OpenAI Services'
      )
    args = parser.parse_args()

    # Modify interpreter according to command line flags
    if args.yes:
        interpreter.auto_run = True
    if args.fast:
        interpreter.model = "gpt-3.5-turbo"
    if args.local:
        interpreter.local = True
    if args.debug:
        interpreter.debug_mode = True
    if args.use_azure:
        interpreter.use_azure = True
        interpreter.local = False

    # Run the chat method
    interpreter.chat()
