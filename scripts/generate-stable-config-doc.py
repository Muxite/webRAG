"""
Generate standardized stable configuration documentation from captured config.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict


def generate_markdown(config_path: str, output_path: str = "STABLE_CONFIG.md"):
    """
    Generate markdown documentation from captured configuration.
    
    :param config_path: Path to captured JSON configuration
    :param output_path: Path to output markdown file
    """
    with open(config_path) as f:
        config = json.load(f)
    
    captured_at = config.get("captured_at", "")
    region = config.get("region", "")
    cluster = config.get("cluster", "")
    
    md_lines = [
        "# Stable Deployment Configuration",
        "",
        f"**Captured At**: {captured_at}",
        f"**Region**: {region}",
        f"**Cluster**: {cluster}",
        "",
        "## Cluster Configuration",
        "",
    ]
    
    cluster_config = config.get("cluster_config", {})
    if "error" not in cluster_config:
        md_lines.extend([
            f"- **Cluster Name**: {cluster_config.get('cluster_name')}",
            f"- **Status**: {cluster_config.get('status')}",
            f"- **Container Insights**: {cluster_config.get('container_insights')}",
            f"- **Running Tasks**: {cluster_config.get('running_tasks_count', 0)}",
            f"- **Active Services**: {cluster_config.get('active_services_count', 0)}",
            "",
        ])
    
    for service_name, service_config in config.get("services", {}).items():
        if "error" in service_config:
            continue
        
        md_lines.extend([
            f"## {service_name} Service",
            "",
        ])
        
        task_def = service_config.get("task_definition_config", {})
        if task_def:
            md_lines.extend([
                f"### Task Definition: {task_def.get('family')}:{task_def.get('revision')}",
                "",
                f"- **CPU**: {task_def.get('cpu')} ({int(task_def.get('cpu', 0)) / 1024:.1f} vCPU)",
                f"- **Memory**: {task_def.get('memory')} MB ({int(task_def.get('memory', 0)) / 1024:.1f} GB)",
                f"- **Network Mode**: {task_def.get('network_mode')}",
                f"- **Registered At**: {task_def.get('registered_at', '')[:19]}",
                "",
                "### Containers",
                "",
            ])
            
            for container in task_def.get("containers", []):
                md_lines.append(f"#### {container.get('name')}")
                md_lines.append("")
                md_lines.append(f"- **Image**: {container.get('image')}")
                md_lines.append(f"- **Essential**: {container.get('essential')}")
                if container.get("cpu"):
                    md_lines.append(f"- **CPU**: {container.get('cpu')}")
                if container.get("memory"):
                    md_lines.append(f"- **Memory**: {container.get('memory')} MB")
                if container.get("memory_reservation"):
                    md_lines.append(f"- **Memory Reservation**: {container.get('memory_reservation')} MB")
                
                health_check = container.get("health_check")
                if health_check:
                    md_lines.append("- **Health Check**:")
                    md_lines.append(f"  - Command: `{' '.join(health_check.get('command', []))}`")
                    md_lines.append(f"  - Interval: {health_check.get('interval')}s")
                    md_lines.append(f"  - Timeout: {health_check.get('timeout')}s")
                    md_lines.append(f"  - Retries: {health_check.get('retries')}")
                    md_lines.append(f"  - Start Period: {health_check.get('start_period')}s")
                
                depends_on = container.get("depends_on", [])
                if depends_on:
                    md_lines.append("- **Depends On**:")
                    for dep in depends_on:
                        md_lines.append(f"  - {dep.get('container_name')} ({dep.get('condition')})")
                
                env_vars = container.get("environment", [])
                if env_vars:
                    md_lines.append("- **Environment Variables**:")
                    for env in sorted(env_vars, key=lambda x: x.get("name", "")):
                        name = env.get("name", "")
                        value = env.get("value", "")
                        if "PASSWORD" in name.upper() or "SECRET" in name.upper() or "KEY" in name.upper():
                            value = "***REDACTED***"
                        md_lines.append(f"  - `{name}`: `{value}`")
                
                md_lines.append("")
        
        md_lines.extend([
            "### Service Configuration",
            "",
            f"- **Status**: {service_config.get('status')}",
            f"- **Desired Count**: {service_config.get('desired_count')}",
            f"- **Running Count**: {service_config.get('running_count')}",
            f"- **Launch Type**: {service_config.get('launch_type')}",
            f"- **Platform Version**: {service_config.get('platform_version')}",
        ])
        
        deployment_config = service_config.get("deployment_configuration", {})
        if deployment_config:
            md_lines.append("- **Deployment Configuration**:")
            md_lines.append(f"  - Maximum Percent: {deployment_config.get('maximum_percent')}")
            md_lines.append(f"  - Minimum Healthy Percent: {deployment_config.get('minimum_healthy_percent')}")
            circuit_breaker = deployment_config.get("deployment_circuit_breaker", {})
            if circuit_breaker:
                md_lines.append(f"  - Circuit Breaker: {circuit_breaker.get('enable', False)}")
        
        if service_config.get("health_check_grace_period_seconds"):
            md_lines.append(f"- **Health Check Grace Period**: {service_config.get('health_check_grace_period_seconds')}s")
        
        load_balancers = service_config.get("load_balancers", [])
        if load_balancers:
            md_lines.append("- **Load Balancers**:")
            for lb in load_balancers:
                md_lines.append(f"  - Target Group: {lb.get('target_group_arn', '').split('/')[-1]}")
                md_lines.append(f"    Container: {lb.get('container_name')}:{lb.get('container_port')}")
        
        service_registries = service_config.get("service_registries", [])
        if service_registries:
            md_lines.append("- **Service Discovery**:")
            for reg in service_registries:
                md_lines.append(f"  - Registry: {reg.get('registry_arn', '').split('/')[-1]}")
                md_lines.append(f"    Port: {reg.get('port')}")
        
        network_config = service_config.get("network_configuration", {}).get("awsvpc_configuration", {})
        if network_config:
            md_lines.append("- **Network Configuration**:")
            md_lines.append(f"  - Subnets: {', '.join(network_config.get('subnets', []))}")
            md_lines.append(f"  - Security Groups: {', '.join(network_config.get('security_groups', []))}")
            md_lines.append(f"  - Assign Public IP: {network_config.get('assign_public_ip')}")
        
        md_lines.append("")
    
    if "target_group" in config:
        tg = config.get("target_group")
        if "error" not in tg:
            md_lines.extend([
                "## Target Group Configuration",
                "",
                f"- **Name**: {tg.get('target_group_name')}",
                f"- **Protocol**: {tg.get('protocol')}",
                f"- **Port**: {tg.get('port')}",
                f"- **VPC**: {tg.get('vpc_id')}",
                "",
                "### Health Check",
                "",
                f"- **Protocol**: {tg.get('health_check_protocol')}",
                f"- **Port**: {tg.get('health_check_port')}",
                f"- **Path**: {tg.get('health_check_path')}",
                f"- **Interval**: {tg.get('health_check_interval_seconds')}s",
                f"- **Timeout**: {tg.get('health_check_timeout_seconds')}s",
                f"- **Healthy Threshold**: {tg.get('healthy_threshold_count')}",
                f"- **Unhealthy Threshold**: {tg.get('unhealthy_threshold_count')}",
                "",
            ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"Generated documentation: {output_path}")


def parse_args():
    """
    Parse CLI arguments.

    :returns: Namespace with config_path and output_path
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate stable config markdown")
    parser.add_argument("config_path", nargs="?", help="Path to captured config JSON")
    parser.add_argument("output_path", nargs="?", default="STABLE_CONFIG.md", help="Output markdown path")
    return parser.parse_args()


def main():
    """
    CLI entrypoint.

    :returns: None
    """
    repo_root = Path(__file__).resolve().parent.parent
    default_config_path = repo_root / "services" / "stable-configs" / "stable-config.json"
    args = parse_args()
    config_path = args.config_path or (
        str(default_config_path) if default_config_path.exists() else "stable-config.json"
    )
    output_path = args.output_path or "STABLE_CONFIG.md"
    
    if not Path(config_path).exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)
    
    generate_markdown(config_path, output_path)


if __name__ == "__main__":
    main()
