from enum import Enum
import os
import atexit
import argparse
from src.utils import get_relative_path


class Colors(Enum):
    """
    Enum for colors used in printing.
    """
    BOLD = "\033[1m"
    RED = "\033[91m"
    STRONGER_RED = "\033[38;5;196m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BOLD_YELLOW = "\033[1;93m"
    BLUE = "\033[94m"
    BOLD_BLUE = "\033[1;94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


class LoggingOptions(Enum):
    """
    Enum for logging options.
    """
    LOG_TO_FILE = "log_to_file"
    LOG_TO_CONSOLE = "log_to_console"
    LOG_TO_BOTH = "log_to_both"
    NO_LOGGING = "no_logging"


class LogLevel(Enum):
    """
    Enum for log levels.
    """
    DEBUG = 10
    PERFORMANCE_UPDATES = 15
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    APPLICATION = 100  # For application-specific logs that are not part of the standard levels

    def __str__(self):
        return self.name    


class LogComponent(Enum):
    """
    Enum for log components.
    """
    ENTITY_LINKER = "Named Entity Linker"
    CLASS_LINKER = "Class Linker"
    GEORELATION_IDENTIFIER = "Geo-relation Identifier"
    RELATION_IDENTIFIER = "KG exploration planner"
    PATH_EXTRACTOR = "KG Explorer"
    QUERY_GENERATOR = "Query Generator"
    KNOWLEDGE_BASE = "Knowledge Base"
    OTHER = "Pythia"
    UTILITIES = "Utilities"

    def __str__(self):
        return self.value 


class LogType(Enum):
    """
    Enum for log types.
    """
    NORMAL = "NORMAL"
    HEADER = "HEADER"
    PROMPT = "PROMPT"
    LLM_RESULT = "LLM_RESULT"
    GOLD = "GOLD"
    
    def __str__(self):
        return self.name    


class PythiaLogger:
    """
    Logger class for Pythia.
    """
    def __init__(self, name: str, directory: str, log_option: LoggingOptions = LoggingOptions.LOG_TO_BOTH):
        self.name = name
        self.directory = directory
        self.full_log_filepath = os.path.join(directory, f"{name}_full.log")
        self.print_log_filepath = os.path.join(directory, f"{name}.log")
        self.log_option = log_option
        if log_option == LoggingOptions.LOG_TO_BOTH or log_option == LoggingOptions.LOG_TO_FILE:
            self.full_log_file, self.print_log_file = self._setup_logging()
            atexit.register(self._cleanup)
        
    def _cleanup(self):
        if self.full_log_file and not self.full_log_file.closed:
            self.full_log_file.flush()
            self.full_log_file.close()
        if self.print_log_file and not self.print_log_file.closed:
            self.print_log_file.flush()
            self.print_log_file.close()

    def _setup_logging(self):
        """
        Set up logging to a file.
        """
        if self.log_option == LoggingOptions.LOG_TO_FILE or self.log_option == LoggingOptions.LOG_TO_BOTH:
            if not os.path.exists(self.full_log_filepath):
                full_logs_file = open(self.full_log_filepath, 'w', buffering=8192)
                full_logs_file.write(f"Log file for {self.name}\n")
            else:
                # exit(f"Log file {self.full_log_filepath} already exists. Please remove it before running the program again.")
                full_logs_file = open(self.full_log_filepath, 'a', buffering=8192)
                
            if not os.path.exists(self.print_log_filepath):
                print_logs_file = open(self.print_log_filepath, 'w', buffering=8192)
                print_logs_file.write(f"Log file for {self.name}\n")
            else:
                # exit(f"Log file {self.print_log_filepath} already exists. Please remove it before running the program again.")
                print_logs_file = open(self.print_log_filepath, 'a', buffering=8192)
            
            return full_logs_file, print_logs_file
    
    def log(self, message: str, log_component: LogComponent, log_level: LogLevel, log_type: LogType):
        """
        Log a message to the log file.
        """
        if self.log_option == LoggingOptions.LOG_TO_FILE or self.log_option == LoggingOptions.LOG_TO_BOTH:
            self.full_log_file.write(f"[{log_level}] - [{log_type}]\n")
            self.full_log_file.write(f"[{log_component}]: {message}\n")
        if self.log_option == LoggingOptions.LOG_TO_CONSOLE or self.log_option == LoggingOptions.LOG_TO_BOTH:
            if print_log_level.value > log_level.value and log_level.value < LogLevel.APPLICATION.value: # skip logging if the log level is lower than the print log level
                return
            color = Colors.RESET
            if log_type == LogType.HEADER:
                color = Colors.BOLD
            elif log_type == LogType.PROMPT:
                color = Colors.GREEN
            elif log_type == LogType.LLM_RESULT:
                color = Colors.CYAN
            elif log_type == LogType.GOLD:
                color = Colors.YELLOW
            elif log_level == LogLevel.WARNING:
                color = Colors.BOLD_YELLOW
            elif log_level == LogLevel.ERROR:
                color = Colors.RED
            elif log_level == LogLevel.CRITICAL:
                color = Colors.STRONGER_RED
            print_colored(f"[{log_component}]: {message}", color)
            if self.log_option == LoggingOptions.LOG_TO_FILE or self.log_option == LoggingOptions.LOG_TO_BOTH:
                self.print_log_file.write(f"[{log_component}]: {message}\n")


logger = None
print_log_level = LogLevel.PERFORMANCE_UPDATES


def create_logger(name: str, dir: str, log_option: LoggingOptions = LoggingOptions.LOG_TO_BOTH, log_level: LogLevel = LogLevel.INFO) -> PythiaLogger:
    """
    Get a logger instance.
    
    :param name: Name of the logger.
    :param log_option: Logging option.
    :return: PythiaLogger instance.
    """
    global logger
    if logger is None:
        logger = PythiaLogger(name, dir, log_option)
        global print_log_level
        print_log_level = log_level
    return logger

def create_console_logger(log_level: LogLevel = LogLevel.INFO) -> None:
    """
    Create a console logger only.
    
    :param log_level: Logging level.
    """
    global logger
    if logger is None:
        logger = PythiaLogger("console_logger", ".", LoggingOptions.LOG_TO_CONSOLE)
        global print_log_level
        print_log_level = log_level


def log(message: str, log_component: LogComponent, log_level: LogLevel = LogLevel.INFO, log_type: LogType = LogType.NORMAL):
    """
    Log a message using the global logger.
    
    :param message: Message to log.
    :param log_level: Log level.
    """
    global logger
    if logger is None:
        raise ValueError("Logger is not initialized. Please create a logger instance first.")
    logger.log(message, log_component, log_level, log_type)


def print_colored(text, color: Colors):
    """
    Print text in a specific color.
    """
    print(f"{color.value}{text}{Colors.RESET.value}")


def print_result(text, color: Colors):
    """
    Print result in a specific color.
    """
    print()
    print("#" * 20)
    print(f"{color.value}{text}{Colors.RESET.value}")
    print("#" * 20)
    print()


def logging_fill_parse_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup: 
    logging_group = parser.add_argument_group("Logging Settings")
    logging_group.add_argument("--logs_dir", type=str, required=True,
                            help="Directory where logs will be saved")
    logging_group.add_argument("--log_level", type=str, required=False, default="performance_updates",
                               help="Logging level: debug, info, performance_updates, warning, error (default: performance_updates)")
    logging_group.add_argument("--log_to_file", type=bool, required=False, default=True,
                               help="If set, logs will be saved to a file in the logs_dir")
    logging_group.add_argument("--log_to_console", type=bool, required=False, default=True,
                               help="If set, logs will be printed to the console")
    return logging_group


def logging_set_from_args(args) -> None:
    if args.log_to_file and args.log_to_console:
        log_option = LoggingOptions.LOG_TO_BOTH
    elif args.log_to_file:
        log_option = LoggingOptions.LOG_TO_FILE
    elif args.log_to_console:
        log_option = LoggingOptions.LOG_TO_CONSOLE
    else:
        log_option = LoggingOptions.NO_LOGGING
    
    if "debug" in args.log_level.lower():
        log_level = LogLevel.DEBUG
    elif "performance_updates" in args.log_level.lower():
        log_level = LogLevel.PERFORMANCE_UPDATES
    elif "info" in args.log_level.lower():
        log_level = LogLevel.INFO
    elif "warning" in args.log_level.lower():
        log_level = LogLevel.WARNING
    elif "error" in args.log_level.lower():
        log_level = LogLevel.ERROR
    elif "critical" in args.log_level.lower():
        log_level = LogLevel.CRITICAL
    
    os.makedirs(get_relative_path(args.logs_dir), exist_ok=True)
    create_logger(args.dataset, args.logs_dir, log_option=log_option, log_level=log_level)