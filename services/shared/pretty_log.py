import logging


def pretty_log(data, logger=None, indents=0):
    """
    Writes a well formatted log message to the console with data that has dicts and or lists.
    :param data: Data made of lists and dicts.
    :param logger: Logger to use or None
    :param indents: How many indents to use at the base.
    """
    logger = logger or logging.getLogger(__name__)
    final = pretty_log_print(data, indents)
    logger.info(final)

def pretty_log_print(data, indents=0):
    indent_str = "    " * indents
    if isinstance(data, list):
        result = []
        for i, item in enumerate(data):
            result.append(f"{indent_str}{i}. {pretty_log_print(item, indents + 1).lstrip()}")
        return "\n".join(result)
    elif isinstance(data, dict):
        result = []
        for key, value in data.items():
            header = f"{indent_str}{str(key).replace('_', ' ').upper()}:"
            value_str = pretty_log_print(value, indents + 1)
            result.append(f"{header}\n{value_str}")
        return "\n".join(result)
    else:
        return f"{indent_str}{str(data)}"

if __name__ == "__main__":
    block = {
        "world" : "Earth",
        "time" : 2025,
        "animals" : [
            {"blue_fish": ["tuna", "sardine", "grouper"]},
            {"birds": ["hawk", "sparrow", "pelican"]},
                    ]
    }
    print(pretty_log_print(block, 1))