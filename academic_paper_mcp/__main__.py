"""Entry point: ``python -m academic_paper_mcp`` serves MCP over stdio."""

from academic_paper_mcp.server import build_server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
