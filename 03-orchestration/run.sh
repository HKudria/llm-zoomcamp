set -a
. ./.env
set +a

export SECRET_GEMINI_API_KEY=$(echo -n $GEMINI_API_KEY | base64)
export SECRET_OPENAI_API_KEY=$(echo -n $OPENAI_API_KEY | base64)
export SECRET_TAVILY_API_KEY=$(echo -n $TAVILY_API_KEY | base64)
export SECRET_ANTHROPIC_BASE_URL=$(echo -n $ANTHROPIC_BASE_URL | base64)
export SECRET_ANTHROPIC_API_KEY=$(echo -n $ANTHROPIC_API_KEY | base64)
export SECRET_ANTHROPIC_MODEL=$(echo -n $ANTHROPIC_MODEL | base64)


docker compose up -d --force-recreate