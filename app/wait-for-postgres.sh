#!/bin/bash
# wait-for-postgres.sh

set -e

host="$1"
port="$2"
shift 2
cmd="$@"

echo "Esperando a que PostgreSQL esté listo en $host:$port ..."

# NS: Loop until db is ready
until nc -z "$host" "$port"; do
  echo "PostgreSQL no está listo, esperando..."
  sleep 1
done

echo "PostgreSQL está listo."

if [ -n "$cmd" ]; then
  echo "Ejecutando comando: $cmd"
  exec $cmd
fi