REF="$(dirname "$0")"
epydemix validate "$REF/sir.yaml"
epydemix run "$REF/sir.yaml" -o outputs/sir.epx > /dev/null
