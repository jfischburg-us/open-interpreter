"""
message_block.py - Display conversational messages.

This module contains the MessageBlock class which is used to display 
conversational messages from users and assistant in the Open Interpreter 
chatbot. 

The MessageBlock displays Markdown-formatted messages in a live-updating 
panel, with a cursor to indicate typing. It handles formatting messages
and distinguishing code blocks from Markdown code formatting.

Key Components:
- MessageBlock: Main class to display message content.
- textify_markdown_code_blocks: Utility to avoid style collision.

"""
import re
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.box import MINIMAL


class MessageBlock:
    """
    MessageBlock - Display conversational messages.

    The MessageBlock class is used to display conversational messages from users
    and the assistant in the Open Interpreter chatbot. It displays Markdown-formatted
    messages in a live-updating panel and includes a cursor to indicate typing.
    This class handles the formatting of messages and distinguishes code blocks
    from Markdown code formatting.

    Attributes:
        live: An instance of Rich's Live class for live-updating the message panel.
        content: The content of the message to be displayed.

    Methods:
        update_from_message(message): Update the message content and refresh the display.
        end(): Finalize the message display.
        refresh(cursor=True): Refresh the message display with optional cursor.

    Usage:
        To use this class, create an instance of MessageBlock and call its methods to
        update and display messages in the chatbot interface.

    Note:
        This class is designed for displaying chatbot messages with proper formatting.

    """

    def __init__(self):
        """
        Initialize a new MessageBlock instance.
        """
        self.live = Live(auto_refresh=False, console=Console())
        self.live.start()
        self.content = ""
        self.output = ""

    def update_from_message(self, message):
        """
        Update the message content from a given message object and refresh the display.

        Args:
            message (dict): A dictionary containing message content.

        Returns:
            None

        """
        self.content = message.get("content", "")
        if self.content:
            self.refresh()

    def end(self):
        """
        Finalize the message display.

        This method should be called to end the message display.

        Returns:
            None

        """
        self.refresh(cursor=False)
        self.live.stop()

    def refresh(self, cursor=True):
        """
        Refresh the message display with optional cursor.

        Args:
            cursor (bool): Indicates whether to display a cursor indicating typing.

        Returns:
            None

        """
        # De-stylize any code blocks in markdown,
        # to differentiate from our Code Blocks
        content = textify_markdown_code_blocks(self.content)

        if cursor:
            content += "█"

        markdown = Markdown(content.strip())
        panel = Panel(markdown, box=MINIMAL)
        self.live.update(panel)
        self.live.refresh()


def textify_markdown_code_blocks(text):
    """
    To distinguish CodeBlocks from markdown code, we simply turn all markdown code
    (like '```python...') into text code blocks ('```text')
    which makes the code black and white.

    Args:
        text (str): The text containing Markdown code blocks.

    Returns:
        str: The modified text with Markdown code blocks replaced by text code blocks.

    """
    replacement = "```text"
    lines = text.split("\n")
    inside_code_block = False

    for i, line in enumerate(lines):
        # If the line matches ``` followed by optional language specifier
        if re.match(r"^```(\w*)$", line.strip()):
            inside_code_block = not inside_code_block

            # If we just entered a code block, replace the marker
            if inside_code_block:
                lines[i] = replacement

    return "\n".join(lines)
