"""
CLI Command / Entry Point for Framework Status Check.
"""

import sys
from core.config import ProjectConfig
from core.system_status import format_system_status, get_system_status


def main() -> None:
    """Executes the status CLI display."""
    config = ProjectConfig()
    print(f"\n{config.project_name} (v{config.version})\n")
    status_str = format_system_status()
    print(status_str)


if __name__ == "__main__":
    main()
