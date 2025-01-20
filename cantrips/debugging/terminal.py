import inspect
import linecache
import os
import socket
import subprocess
import traceback
from multiprocessing import current_process
from tqdm import tqdm

bcolors = {
    "PINK": "\033[95m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
}


class UGENT:
    BLUE = "#1E64C8"
    YELLOW = "#FFD200"
    WHITE = "#FFFFFF"
    BLACK = "#000000"
    ORANGE = "#F1A42B"
    RED = "#DC4E28"
    AQUA = "#2D8CA8"
    PINK = "#E85E71"
    SKY = "#8BBEE8"
    LIGHTGREEN = "#AEB050"
    PURPLE = "#825491"
    WARMORANGE = "#FB7E3A"
    TURQUOISE = "#27ABAD"
    LIGHTPURPLE = "#BE5190"
    GREEN = "#71A860"

    COLORS = [
        BLUE,
        YELLOW,
        ORANGE,
        RED,
        AQUA,
        PINK,
        SKY,
        LIGHTGREEN,
        PURPLE,
        WARMORANGE,
        TURQUOISE,
        LIGHTPURPLE,
        GREEN,
    ]

    PRIMARY_COLORS = [BLUE, YELLOW]
    SECONDARY_COLORS = [
        ORANGE,
        RED,
        AQUA,
        PINK,
        SKY,
        LIGHTGREEN,
        PURPLE,
        WARMORANGE,
        TURQUOISE,
        LIGHTPURPLE,
        GREEN,
    ]

def hex2rgb(color: str) -> tuple:
    """
    Convert a hexadecimal color string to a BGR (Blue, Green, Red) tuple.

    Args:
        color (str): A hexadecimal color string, starting with '#' and followed
                     by 6 hexadecimal digits (e.g., "#RRGGBB").

    Returns:
        tuple: A tuple of integers representing the BGR values in the range
               0 to 255, e.g., (B, G, R).

    Raises:
        ValueError: If the input is not a valid hexadecimal color string.
    """
    if not isinstance(color, str) or not color.startswith("#") or len(color) != 7:
        raise ValueError("Invalid hex color format. Must be in the format '#RRGGBB'.")

    # Extract the red, green, and blue components from the hex string
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)

    # Return BGR tuple
    return (r, g, b)


def pretty_string(message: str, color=None, bold=False, underline=False):
    """
    add color and effects to string
    :param message:
    :param color:
    :param bold:
    :param underline:
    :return:
    """
    ou = message
    if color:
        ou = bcolors[color] + message + "\033[0m"
    if bold:
        ou = "\033[1m" + ou + "\033[0m"
    if underline:
        ou = "\033[4m" + ou + "\033[0m"
    return ou


def poem(string, maxlen=23):
    if len(string) > maxlen:
        return string[:maxlen-3] + "..."
    else:
        return string + " " * (maxlen - len(string))


def pyout(*message, color="PINK"):
    """
    Print message preceded by traceback, and now including the argument names.
    :param message: The message(s) to print.
    """

    frame = inspect.currentframe().f_back
    line = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()

    # Initial approach to find the start of the pyout call
    start_index = line.find("pyout(")
    if start_index == -1:
        arg_str = ""
    else:
        # Count parentheses to find the correct closing one
        count = 0
        for i, char in enumerate(line[start_index:]):
            if char == "(":
                count += 1
            elif char == ")":
                count -= 1
                if count == 0:
                    # Found the matching closing parenthesis
                    arg_str = line[
                        start_index + 6 : start_index + i
                    ]  # Exclude "pyout(" and ")"
                    break
        else:
            # Didn't find a matching closing parenthesis (unlikely unless the line is malformed)
            arg_str = line[start_index + 6 :]

    trace = traceback.extract_stack()[-2]
    fname = trace.filename.replace(os.path.abspath(os.curdir), "...")
    trace_info = f"{fname}: {trace.name}(...)"

    # Print the argument string, if found
    if arg_str:
        tqdm.write(pretty_string(trace_info, color, bold=True))
        tqdm.write(
            pretty_string(f"ln{trace.lineno}   ", color, bold=True)
            + pretty_string(f"{arg_str} = ...", color, bold=False)
        )
    else:
        tqdm.write(pretty_string(f"{trace_info} - ln{trace.lineno}", color, bold=True))

    # Finally, print the message, if any
    if message:
        message_text = " ".join(str(m) for m in message)
        tqdm.write(message_text)


def pbar(iterable, desc="", leave=False, total=None, disable=False):
    # return iterable
    host = socket.gethostname()

    if host in ("AM", "kat", "gorilla"):
        return tqdm(
            iterable,
            desc=poem(desc),
            leave=leave,
            total=total,
            disable=(current_process().name != "MainProcess"),
        )
    else:
        return iterable


def prog():
    # Get the caller's frame
    caller_frame = inspect.currentframe().f_back
    # Get the filename and line number of the caller
    filename = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno

    # Get the previous line from the file
    previous_line = linecache.getline(filename, line_number + 1).strip()

    print(f"ln {line_number}: {previous_line}")

    # Clear the cache and delete the frame to help with garbage collection
    linecache.clearcache()
    del caller_frame

def pysend(*message):
    message = " ".join(str(m) for m in message)
    trace = traceback.extract_stack()[-2]

    fname = trace.filename.replace(os.path.abspath(os.curdir), "...")

    trace = f"{fname}: {trace.name}(...) - ln{trace.lineno}"

    subprocess.Popen(["notify-send", trace, message])
