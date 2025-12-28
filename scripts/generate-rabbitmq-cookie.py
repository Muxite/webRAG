import secrets
import string

"""
Generate a 255-character alphanumeric string for RABBITMQ_ERLANG_COOKIE.
This can be manually pasted into keys.env for use in AWS Secrets Manager.
"""

def generate_rabbitmq_cookie(length=255):
    """Generate a random alphanumeric string of specified length."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

if __name__ == "__main__":
    cookie = generate_rabbitmq_cookie()
    print(cookie)

