"""
This module initializes the 'interpreter' package and provides
a convenient way for users to access the 'Interpreter' class instance.

By assigning an instance of 'Interpreter' to the 'interpreter'
module, users can easily access the chatbot functionality without
needing to create an instance themselves.
"""
import sys
from .interpreter import Interpreter

# This is done so when users `import interpreter`,
# they get an instance of interpreter:

sys.modules["interpreter"] = Interpreter()

# **This is a controversial thing to do,**
# because perhaps modules ought to behave like modules.

# But I think it saves a step, removes friction, and looks good.

#     ____                      ____      __                            __
#    / __ \____  ___  ____     /  _/___  / /____  _________  ________  / /____  _____
#   / / / / __ \/ _ \/ __ \    / // __ \/ __/ _ \/ ___/ __ \/ ___/ _ \/ __/ _ \/ ___/
#  / /_/ / /_/ /  __/ / / /  _/ // / / / /_/  __/ /  / /_/ / /  /  __/ /_/  __/ /
#  \____/ .___/\___/_/ /_/  /___/_/ /_/\__/\___/_/  / .___/_/   \___/\__/\___/_/
#      /_/                                         /_/
