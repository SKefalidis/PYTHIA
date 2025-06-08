
import os
import re
import sys
import yaml
import argparse
import flatdict

class CONFIG:
    """Engine configuration singleton class."""

    _instance = None

    def __new__(cls, args: argparse.Namespace = None):
        if cls._instance is None:
            if args is None:
                print("WARNING: CONFIG is being initialized without arguments. Command-line arguments will be ignored.", file=sys.stderr)
            
            cls._instance = super(CONFIG, cls).__new__(cls)

            # Default configuration
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(root_dir, "config.yaml")
            
            # If the parser is provided, check for --config argument
            if args is not None:
                if getattr(args, "config", None) is not None:
                    config_path = args.config
                    
            # 1. Environment Variable Resolver Logic
            env_pattern = re.compile(r'env\((?P<var>[^:]+):?(?P<default>[^)]*)\)')

            def env_constructor(loader, node):
                value = loader.construct_scalar(node)
                match = env_pattern.match(value)
                if match:
                    var_name = match.group("var")
                    default = match.group("default")
                    if default.lower() == "null":
                        default = None
                    return os.getenv(var_name, default)
                return value

            yaml.SafeLoader.add_implicit_resolver("!env", env_pattern, None)
            yaml.SafeLoader.add_constructor("!env", env_constructor)

            # 2. Robust Flattening Function (Replaces flatdict to avoid collisions)
            def flatten_config(d, parent_key='', sep='_'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_config(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            # --- Integrated into your Loading Logic ---

            # Load configuration from file
            if not os.path.exists(config_path):
                print(f'ERROR: configuration file not found: {config_path}', file=sys.stderr)
                sys.exit(1)

            try:
                with open(config_path, "r") as f:
                    # Load with environment variable resolution
                    raw_config = yaml.safe_load(f)
                    
                    # Use the manual flattening function instead of flatdict
                    # This prevents the "Assignment to invalid type" error
                    cls._instance.config = flatten_config(raw_config, sep="_")
                    
                    print("Successfully loaded config:", cls._instance.config)

            except Exception as e:
                print(f'ERROR: failed to read configuration file {config_path}: {e}', file=sys.stderr)
                sys.exit(1)

            # Override configuration file with environment variables
            if args is not None:
                for key, value in vars(args).items():
                    if value is not None:
                        cls._instance.config[key] = value
                        print(f'Overriding config key "{key}" with command-line argument value: {value!r}')

        return cls._instance

    def get(self, key: str, default=None):
        if key not in self._instance.config:
            print(f'WARNING: configuration key "{key}" not found. Returning default value: {default!r}', file=sys.stderr)
        return self._instance.config.get(key, default)
    
    def set(self, key: str, value):
        self._instance.config[key] = value
    
    def print_config(self):
        for key, value in self._instance.config.items():
            print(f"{key}: {value}")
            
    def get_litellm_model_endpoint(self):
        """Returns the Litellm endpoint URL based on the configuration."""
        return self.get("llm_provider") + "/" + self.get("llm_model")


if __name__ == "__main__":
    from src.elelem.provider import *
    parser = argparse.ArgumentParser(description="Test CONFIG class")
    parser.add_argument("--test_param", type=str, help="A test parameter")
    ProviderFactory.fill_parse_args(parser)
    args = parser.parse_args()
    config = CONFIG(args)
    config.print_config()
    print(config.get("test_param"))
    print(config.get("index_dir"))