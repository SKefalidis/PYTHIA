# Usage: source setup_terminal.sh <AWS_BEARER_TOKEN_BEDROCK> <OPENAI_API_KEY>
# Or: . setup_terminal.sh <aws_token> <openai_key>

if [ "$#" -ne 2 ]; then
    echo "Usage: source $0 <AWS_BEARER_TOKEN_BEDROCK> <OPENAI_API_KEY>"
    # If sourced, return; if executed, exit.
    return 1 2>/dev/null || exit 1
fi

export AWS_BEARER_TOKEN_BEDROCK="$1"
export OPENAI_API_KEY="$2"

echo "Exported AWS_BEARER_TOKEN_BEDROCK and OPENAI_API_KEY."