from mlflow import MlflowClient

from src.config import load_config


def main() -> None:
    config = load_config()

    model_name = config["mlflow"]["registered_model_name"]
    client = MlflowClient()

    versions = client.search_model_versions(f"name = '{model_name}'")

    if not versions:
        raise ValueError(f"No versions found for model: {model_name}")

    latest_version = max(versions, key=lambda version: int(version.version))

    client.set_registered_model_alias(
        name=model_name,
        alias="candidate",
        version=latest_version.version,
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=latest_version.version,
    )

    print(f"Set candidate alias -> version {latest_version.version}")
    print(f"Set champion alias -> version {latest_version.version}")


if __name__ == "__main__":
    main()
