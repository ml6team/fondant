import json

from utils.emails_ip_addresses_detection import detect_email_addresses
from utils.keys_detection import detect_keys


def postprocess_secrets(secrets):
    """Postprocess the secrets found by the scan_secrets function."""
    if secrets:
        matches = json.dumps(secrets)
        has_secrets = True
    else:
        matches = json.dumps([])
        has_secrets = False
    return matches, has_secrets


def scan_pii(text, key_detector="other"):
    """Scan a piece of code to detect PII
    This add 3 columns to the dataset:
    - secrets: (list) of secrets/PII found
    - has_secrets: (bool) whether the example contains secrets/PII
    - num_secrests (int) number of secrets.
    """
    secrets = []
    if key_detector == "regex":
        # use a regex to detect keys + emails + ips
        secrets = secrets + detect_email_addresses(
            text, tag_types={"KEY", "EMAIL", "IP_ADDRESS"},
        )
    else:
        # detect emails and ip addresses with regexes
        secrets = secrets + detect_email_addresses(
            text, tag_types={"EMAIL", "IP_ADDRESS"},
        )
        # for keys use detect-secrets tool
        secrets = secrets + detect_keys(text)
    # to add this as new columns to datasets we need the same number of samples in each row
    # we save secrets as json strings instead of lists
    matches, has_secrets = postprocess_secrets(secrets)

    return matches, has_secrets, len(secrets)
