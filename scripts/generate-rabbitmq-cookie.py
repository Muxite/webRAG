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

if __name__ == "__main__":
    cookie = generate_rabbitmq_cookie()
    print(cookie)

