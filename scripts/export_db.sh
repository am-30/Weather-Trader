#!/bin/bash
# =============================================================================
# Export Replit database for migration to DigitalOcean VM.
# Run this from the Replit shell BEFORE cancelling your Replit subscription.
#
# Usage:
#   bash scripts/export_db.sh
#
# Output: replit_db_export.sql (in the current directory)
# =============================================================================
set -euo pipefail

OUTPUT="replit_db_export.sql"

if [ -z "${DATABASE_URL:-}" ]; then
    echo "ERROR: DATABASE_URL environment variable is not set."
    echo "       Make sure you're running this from the Replit shell."
    exit 1
fi

echo "Exporting database to $OUTPUT ..."
pg_dump "$DATABASE_URL" > "$OUTPUT"

ROWS=$(wc -l < "$OUTPUT")
echo "Done. Export file: $OUTPUT ($ROWS lines)"
echo ""
echo "============================================================"
echo " Next steps to import on your DigitalOcean VM:"
echo "============================================================"
echo ""
echo "1. Download replit_db_export.sql from Replit."
echo "   (Use the Replit file browser or scp)"
echo ""
echo "2. Upload to your VM:"
echo "   scp replit_db_export.sql root@<your-droplet-ip>:/tmp/"
echo ""
echo "3. Import on the VM:"
echo "   sudo -u trader psql kalshi_trader < /tmp/replit_db_export.sql"
echo ""
echo "4. Then run: bash /home/trader/kalshi-weather-trader/deploy/install_services.sh"
echo ""
