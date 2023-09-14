"""
code_block.py - Display code and outputs in different languages.

This module contains the CodeBlock class, which is used to display code and outputs
in different programming languages in a live-updating panel. It provides functionality
to highlight code syntax, handle code execution, and display code outputs.

Key Components:

- CodeBlock: Main class for displaying and managing code and outputs.

"""
from rich.live import Live
from rich.panel import Panel
from rich.box import MINIMAL
from rich.syntax import Syntax
from rich.table import Table
from rich.console import Group
from rich.console import Console


class CodeBlock:
    """
    Code Blocks display code and outputs in different languages.

    The CodeBlock class is used to display code and outputs in various programming
    languages. It provides features for syntax highlighting, code execution, and
    live-updating display. This class is designed for interactive code demonstrations
    and tutorials.

    Attributes:
        language (str): The programming language associated with the code.
        output (str): The output generated by the executed code.
        code (str): The code to be displayed and executed.
        active_line (int): The line number of the currently active code line.
        live: An instance of Rich's Live class for live-updating the code panel.

    Methods:
        update_from_message(message): Update code and display based on a message.
        end(): Finalize the code display.
        refresh(cursor=True): Refresh the code display with optional cursor.

    Usage:
        To use this class, create an instance of CodeBlock and call its methods to
        update and display code and outputs in different programming languages.

    Note:
        This class is intended for interactive code presentation and execution.

    """

    def __init__(self):
        """
        Initialize a new CodeBlock instance.
        """
        # Define these for IDE auto-completion
        self.language = ""
        self.output = ""
        self.code = ""
        self.active_line = None

        self.live = Live(
            auto_refresh=False, console=Console(), vertical_overflow="visible"
        )
        self.live.start()

    def update_from_message(self, message):
        """
        Update code and display based on a message.

        Args:
            message (dict): A dictionary containing code and output information.

        Returns:
            None

        """
        if (
            "function_call" in message
            and "parsed_arguments" in message["function_call"]
        ):
            parsed_arguments = message["function_call"]["parsed_arguments"]

            if parsed_arguments is not None:
                self.language = parsed_arguments.get("language")
                self.code = parsed_arguments.get("code")

                if self.code and self.language:
                    self.refresh()

    def end(self):
        """
        Finalize the code display.

        This method should be called to end the code display.

        Returns:
            None

        """
        self.refresh(cursor=False)
        # Destroys live display
        self.live.stop()

    def refresh(self, cursor=True):
        """
        Refresh the code display with optional cursor.

        Args:
            cursor (bool): Indicates whether to display a cursor indicating typing.

        Returns:
            None

        """
        # Get code, return if there is none
        code = self.code
        if not code:
            return

        # Create a table for the code
        code_table = Table(
            show_header=False, show_footer=False, box=None, padding=0, expand=True
        )
        code_table.add_column()

        # Add cursor
        if cursor:
            code += "█"

        # Add each line of code to the table
        code_lines = code.strip().split("\n")
        for i, line in enumerate(code_lines, start=1):
            if i == self.active_line:
                # This is the active line, print it with a white background
                syntax = Syntax(
                    line, self.language, theme="bw", line_numbers=False, word_wrap=True
                )
                code_table.add_row(syntax, style="black on white")
            else:
                # This is not the active line, print it normally
                syntax = Syntax(
                    line,
                    self.language,
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True,
                )
                code_table.add_row(syntax)

        # Create a panel for the code
        code_panel = Panel(code_table, box=MINIMAL, style="on #272722")

        # Create a panel for the output (if there is any)
        if self.output == "" or self.output == "None":
            output_panel = ""
        else:
            output_panel = Panel(self.output, box=MINIMAL, style="#FFFFFF on #3b3b37")

        # Create a group with the code table and output panel
        group = Group(
            code_panel,
            output_panel,
        )

        # Update the live display
        self.live.update(group)
        self.live.refresh()
