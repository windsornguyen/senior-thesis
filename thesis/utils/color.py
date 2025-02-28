from colorama import Fore, Style, init

init(autoreset=True)

class Color:
    """
    Centralized color management for terminal output.
    Provides preset color schemes and utility methods for terminal coloring.

    If initialized with `enabled=False`, all color/style attributes become empty strings,
    effectively disabling color printing.
    """

    def __init__(self, enabled: bool = True):
        """
        Instantiate the Color manager. If `enabled` is False, all attributes
        return empty strings instead of ANSI codes.
        """
        # Basic colors
        self.BLACK = Fore.BLACK if enabled else ""
        self.RED = Fore.RED if enabled else ""
        self.GREEN = Fore.GREEN if enabled else ""
        self.YELLOW = Fore.YELLOW if enabled else ""
        self.BLUE = Fore.BLUE if enabled else ""
        self.MAGENTA = Fore.MAGENTA if enabled else ""
        self.CYAN = Fore.CYAN if enabled else ""
        self.WHITE = Fore.WHITE if enabled else ""

        # Bright variants
        self.BRIGHT_BLACK = Fore.LIGHTBLACK_EX if enabled else ""
        self.BRIGHT_RED = Fore.LIGHTRED_EX if enabled else ""
        self.BRIGHT_GREEN = Fore.LIGHTGREEN_EX if enabled else ""
        self.BRIGHT_YELLOW = Fore.LIGHTYELLOW_EX if enabled else ""
        self.BRIGHT_BLUE = Fore.LIGHTBLUE_EX if enabled else ""
        self.BRIGHT_MAGENTA = Fore.LIGHTMAGENTA_EX if enabled else ""
        self.BRIGHT_CYAN = Fore.LIGHTCYAN_EX if enabled else ""
        self.BRIGHT_WHITE = Fore.LIGHTWHITE_EX if enabled else ""

        # Style options
        self.BOLD = Style.BRIGHT if enabled else ""
        self.DIM = Style.DIM if enabled else ""
        self.NORMAL = Style.NORMAL if enabled else ""
        self.RESET = Style.RESET_ALL if enabled else ""

        # Semantic color schemes
        # Note we still combine them here—if `enabled=False`, they become empty anyway.
        self.SUCCESS = self.GREEN + self.BOLD
        self.ERROR = self.RED + self.BOLD
        self.WARNING = self.YELLOW + self.BOLD
        self.INFO = self.CYAN
        self.DEBUG = self.MAGENTA
        self.METRIC = self.BLUE
        self.TIME = self.WHITE + self.DIM
        self.VALUE = self.BOLD + self.WHITE

        # Keep a flag for other internal logic if needed
        self.enabled = enabled

    def combine(self, *styles: str) -> str:
        """
        Combine multiple color and style attributes.
        If not enabled, this is effectively empty or minimal.
        """
        return "".join(styles)

    def wrap(self, text: str, *styles: str) -> str:
        """
        Wrap text with specified colors/styles and auto-reset.
        If color is disabled, returns the original text unchanged.
        """
        combined = self.combine(*styles)
        return f"{combined}{text}{self.RESET}" if self.enabled else text

    def success(self, text: str) -> str:
        """Format text as success message."""
        return self.wrap(text, self.SUCCESS)

    def error(self, text: str) -> str:
        """Format text as error message."""
        return self.wrap(text, self.ERROR)

    def warning(self, text: str) -> str:
        """Format text as warning message."""
        return self.wrap(text, self.WARNING)

    def info(self, text: str) -> str:
        """Format text as info message."""
        return self.wrap(text, self.INFO)

    def debug(self, text: str) -> str:
        """Format text as debug message."""
        return self.wrap(text, self.DEBUG)

    def metric(self, name: str, value: str) -> str:
        """Format a metric name and value pair."""
        return f"{self.wrap(name, self.METRIC)}: {self.wrap(str(value), self.VALUE)}"

    def time(self, text: str) -> str:
        """Format text as timestamp."""
        return self.wrap(text, self.TIME)


# Provide a ready-made "no-color" instance for convenience.
NoColor = Color(enabled=False)
