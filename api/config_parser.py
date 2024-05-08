import json
from botocore.exceptions import ClientError


class ConfigParser:

    def __init__(self, session):
        self.ssm_client = session.client("ssm")

    def load_config(self, parameter_name: str):
        try:
            response = self.ssm_client.get_parameter(
                Name=parameter_name, WithDecryption=True
            )
            key = response["Parameter"]["Value"]
            print("openai key---------->", key)
            # Validate and return the configuration {"key":"xxxxxxxxxxxx"}
            return key
        except ClientError as e:
            raise Exception(f"Failed to fetch parameter: {e}")
