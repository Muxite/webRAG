from dotenv import dotenv_values
import json


def write_dict(input_dict):
    """
    Write dictionary to JSON file and return file:// URI.
    
    :param input_dict: Dictionary to write.
    :returns: File URI string.
    """
    with open("secrets.json", "w") as f:
        f.write(json.dumps(input_dict))

    return "file://secrets.json"


def format_json_array(items, indent_level=4):
    """
    Format a list of items as a JSON array string with indentation.
    
    :param items: List of items to format.
    :param indent_level: Number of spaces for indentation.
    :returns: Formatted JSON array string.
    """
    if not items:
        return "[]"

    indent = " " * indent_level
    result = "["
    for i, item in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        item_str = json.dumps(item, indent=4)

        indented_item = "\n".join(indent + " " * 4 + line for line in item_str.split("\n"))
        result += indented_item + comma + "\n"
    result += indent + "]"
    return result


def main():
    """
    Main entry point for generating ECS task definition configuration.
    Prompts for secret name, region, account ID, and ARN, then outputs
    formatted secrets and environment variable configuration.
    """
    keys_env = dict(dotenv_values("services/keys.env"))
    env = dict(dotenv_values("services/.env"))

    name = str(input("Enter secret name: "))
    region = str(input("Region: "))
    account_id = str(input("AWS Account ID: "))

    print("\n" + "=" * 32)
    print("\nCOMMANDS TO CREATE OR UPDATE SECRETS:")
    print("=" * 32 + "\n")
    print(f"aws secretsmanager create-secret --name {name} --secret-string {write_dict(keys_env)} --region {region}")
    print(f"\naws secretsmanager update-secret --secret-id {name} --secret-string {write_dict(keys_env)} --region {region}")

    print("\n" + "=" * 32)
    print("TASK DEFINITION CONFIG (paste into containerDefinitions):")
    print("=" * 32 + "\n")
    arn = str(input("Enter secret ARN: "))
    secrets_json = []
    for key in keys_env.keys():
        secret_entry = {
            "name": key,
            "valueFrom": f"arn:aws:secretsmanager:{region}:{account_id}:secret:{name}-{arn}:{key}::"
        }
        secrets_json.append(secret_entry)

    env_json = []
    for key, value in env.items():
        env_entry = {
            "name": key,
            "value": value
        }
        env_json.append(env_entry)

    print('"secrets": [')
    for i, item in enumerate(secrets_json):
        comma = "," if i < len(secrets_json) - 1 else ""
        print(f'                {{')
        print(f'                    "name": "{item["name"]}",')
        print(f'                    "valueFrom": "{item["valueFrom"]}"')
        print(f'                }}{comma}')
    print('            ],')


    print('"environment": [')
    for i, item in enumerate(env_json):
        comma = "," if i < len(env_json) - 1 else ""
        print(f'                {{')
        print(f'                    "name": "{item["name"]}",')
        print(f'                    "value": "{item["value"]}"')
        print(f'                }}{comma}')
    print('            ]')

    print("\n" + "=" * 32)
    print(f"Total secrets: {len(keys_env)}")
    print(f"Total environment variables: {len(env)}")
    print("=" * 32)


if __name__ == "__main__":
    main()