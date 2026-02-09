"""
Deployment mode enumeration.
"""
from enum import Enum


class DeploymentMode(Enum):
    """
    Deployment mode enumeration.
    
    SINGLE: All containers in a single ECS service
    AUTOSCALE: Separate gateway and agent services with autoscaling
    """
    SINGLE = "single"
    AUTOSCALE = "autoscale"
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_string(cls, value: str):
        """
        Create DeploymentMode from string.
        
        :param value: String value ("single" or "autoscale")
        :returns: DeploymentMode enum value
        :raises ValueError: If value is not a valid mode
        """
        value_lower = value.lower()
        for mode in cls:
            if mode.value == value_lower:
                return mode
        raise ValueError(f"Invalid deployment mode: {value}. Must be 'single' or 'autoscale'")


def main():
    """
    Main entry point.
    """
    modes = ", ".join([mode.value for mode in DeploymentMode])
    print(f"Available modes: {modes}")


if __name__ == "__main__":
    main()