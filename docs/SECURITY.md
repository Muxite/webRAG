# SECURITY

This document explains how API keys are handled by the Euglena Gateway and how to set up development/testing keys safely.

## Overview

The Gateway protects all endpoints using an API key sent in the `X-API-Key` HTTP header. A request is authorized if and only if the provided key matches one of the allowed keys known to the server.

Allowed keys can come from:

- A primary key from an environment variable.
- A short list of special keys stored in a local file (for admin/dev testing only).
- A generated key during automated tests.

## Environment variables

- `GATEWAY_API_KEY` — Primary API key. If set, this key is accepted.
- `TEST_MODE` — When set to `1`, the server will generate a temporary API key for tests if `GATEWAY_API_KEY` is not set. The generated key is placed in the environment so tests can read it.
- `GATEWAY_SPECIAL_KEYS_FILE` — Optional path to a file containing additional allowed keys (one per line). If not set, the application looks for a file named `special_api_keys.txt` in the working directory.

## Special keys file

For admin/dev testing, you may maintain a short list of extra keys in a local file. The Gateway will accept any key listed in this file in addition to the primary key.

Defaults:

- Default filename: `special_api_keys.txt` (in the project root) unless `GATEWAY_SPECIAL_KEYS_FILE` is set.
- File format: UTF-8 text, one key per line.
- Blank lines and lines starting with `#` are ignored.

Example (`special_api_keys.txt`):

```
# Local admin/testing keys (DO NOT COMMIT REAL SECRETS)
# Each non-empty, non-comment line is a valid key
ADMIN_TEST_KEY_1
ADMIN_TEST_KEY_2
```

Important: This file is intended for local use only and should be added to `.gitignore` to avoid committing secrets. The repository does not rely on the presence of this file; if it is missing or unreadable, only the primary key (and test-generated key, when applicable) will be accepted.

## Client usage

Include the API key with each request:

```
X-API-Key: <your key>
```