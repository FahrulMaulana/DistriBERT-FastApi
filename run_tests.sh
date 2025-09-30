#!/bin/bash

# Test Runner for Hybrid QA Chatbot System
# Usage: ./run_tests.sh [service_url]

set -e

# Configuration
SERVICE_URL=${1:-"http://localhost:8000"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_hybrid_system.py"

echo "🚀 Hybrid QA Chatbot Test Runner"
echo "=================================="
echo "Service URL: $SERVICE_URL"
echo "Test Script: $TEST_SCRIPT"
echo ""

# Check if Python test script exists
if [ ! -f "$TEST_SCRIPT" ]; then
    echo "❌ Test script not found: $TEST_SCRIPT"
    exit 1
fi

# Check if service is running
echo "🔍 Checking service health..."
if curl -s "$SERVICE_URL/health" > /dev/null; then
    echo "✅ Service is responding"
else
    echo "❌ Service is not responding at $SERVICE_URL"
    echo "   Make sure the service is running with:"
    echo "   uvicorn main:app --host 0.0.0.0 --port 8000"
    exit 1
fi

# Install test dependencies if needed
echo "📦 Checking test dependencies..."
if ! python3 -c "import requests" 2>/dev/null; then
    echo "Installing requests..."
    pip install requests
fi

# Run the test suite
echo ""
echo "🧪 Running comprehensive test suite..."
echo "======================================"

python3 "$TEST_SCRIPT" "$SERVICE_URL"
TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "🎉 All tests passed! System is ready for production."
else
    echo "❌ Some tests failed. Please review the output above."
fi

exit $TEST_RESULT
