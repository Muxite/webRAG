"""
Generate a RabbitMQ Erlang cookie value.

:returns: None
"""
import secrets
import string


def generate_rabbitmq_cookie(length=255):
    """
    Generate a random alphanumeric string for RABBITMQ_ERLANG_COOKIE.
    
    :param length: Length of the cookie string.
    :returns: Random alphanumeric string.
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def parse_args():
    """
    Parse CLI arguments.

    :returns: Namespace with length
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate RabbitMQ Erlang cookie")
    parser.add_argument("--length", type=int, default=255, help="Cookie length")
    return parser.parse_args()


def main():
    """
    CLI entrypoint.

    :returns: None
    """
    args = parse_args()
    cookie = generate_rabbitmq_cookie(length=args.length)
    print(cookie)


if __name__ == "__main__":
    main()

