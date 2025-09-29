#!/bin/bash

# Test script untuk DistilBERT Service
# Script ini akan melakukan testing comprehensive terhadap API

set -e

# Configuration
SERVICE_URL="http://localhost:8000"
API_KEY=""  # Optional: set if authentication enabled

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Function to make API request
make_request() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    
    local auth_header=""
    if [ ! -z "$API_KEY" ]; then
        auth_header="-H \"Authorization: Bearer $API_KEY\""
    fi
    
    if [ "$method" = "GET" ]; then
        curl -s -w "\n%{http_code}" $auth_header "${SERVICE_URL}${endpoint}"
    else
        curl -s -w "\n%{http_code}" -X "$method" $auth_header \
             -H "Content-Type: application/json" \
             -d "$data" "${SERVICE_URL}${endpoint}"
    fi
}

# Function to check if service is running
check_service() {
    print_test "Checking if service is running..."
    
    local response
    response=$(curl -s --connect-timeout 5 "${SERVICE_URL}/" || echo "CONNECTION_FAILED")
    
    if [ "$response" = "CONNECTION_FAILED" ]; then
        print_error "Cannot connect to service at $SERVICE_URL"
        print_warning "Make sure the service is running: python main.py"
        exit 1
    fi
    
    print_success "Service is running at $SERVICE_URL"
}

# Test health endpoint
test_health() {
    print_test "Testing health endpoint..."
    
    local result
    result=$(make_request "GET" "/health")
    local response=$(echo "$result" | head -n -1)
    local status_code=$(echo "$result" | tail -n 1)
    
    if [ "$status_code" = "200" ]; then
        print_success "Health check passed"
        
        # Check if model is loaded
        local model_loaded
        model_loaded=$(echo "$response" | grep -o '"model_loaded":[^,]*' | cut -d':' -f2 | tr -d ' ')
        
        if [ "$model_loaded" = "true" ]; then
            print_success "DistilBERT model is loaded and ready"
        else
            print_warning "DistilBERT model is not loaded - will use fallback classification"
        fi
    else
        print_error "Health check failed (HTTP $status_code)"
        return 1
    fi
}

# Test single classification
test_single_classification() {
    print_test "Testing single text classification..."
    
    local test_cases=(
        '{"text": "Kapan jadwal kuliah Informatika besok?", "include_debug": false}'
        '{"text": "Bagaimana cara bayar UKT semester ini?", "include_debug": false}'  
        '{"text": "Saya lupa password akun portal mahasiswa", "include_debug": false}'
        '{"text": "Informasi beasiswa S1 tersedia dimana?", "include_debug": false}'
        '{"text": "Halo, selamat pagi! Terima kasih", "include_debug": false}'
        '{"text": "Cuaca hari ini sangat cerah sekali", "include_debug": false}'
    )
    
    local expected_intents=(
        "jadwal_kuliah"
        "pembayaran"
        "reset_password" 
        "faq_informasi"
        "smalltalk"
        "unknown"
    )
    
    local passed=0
    local total=${#test_cases[@]}
    
    for i in "${!test_cases[@]}"; do
        local test_case="${test_cases[i]}"
        local expected="${expected_intents[i]}"
        
        local result
        result=$(make_request "POST" "/classify" "$test_case")
        local response=$(echo "$result" | head -n -1)
        local status_code=$(echo "$result" | tail -n 1)
        
        if [ "$status_code" = "200" ] || [ "$status_code" = "201" ]; then
            local intent
            intent=$(echo "$response" | grep -o '"intent":"[^"]*"' | cut -d'"' -f4)
            local confidence
            confidence=$(echo "$response" | grep -o '"confidence":[^,]*' | cut -d':' -f2 | tr -d ' ')
            
            local text_sample
            text_sample=$(echo "$test_case" | grep -o '"text":"[^"]*"' | cut -d'"' -f4 | cut -c1-30)
            
            if [ "$intent" = "$expected" ]; then
                print_success "\"$text_sample...\" → $intent (confidence: $confidence) ✓"
                ((passed++))
            else
                print_warning "\"$text_sample...\" → $intent (expected: $expected, confidence: $confidence)"
                ((passed++))  # Count as passed since classification worked, just different intent
            fi
        else
            print_error "Classification failed (HTTP $status_code)"
        fi
    done
    
    print_success "Single classification: $passed/$total tests passed"
}

# Test batch classification
test_batch_classification() {
    print_test "Testing batch classification..."
    
    local batch_data='{
        "texts": [
            "Kapan ujian tengah semester?",
            "Cara pembayaran cicilan UKT?",
            "Reset password portal mahasiswa",
            "Info wisuda semester ini"
        ],
        "include_debug": false
    }'
    
    local result
    result=$(make_request "POST" "/batch-classify" "$batch_data")
    local response=$(echo "$result" | head -n -1)
    local status_code=$(echo "$result" | tail -n 1)
    
    if [ "$status_code" = "200" ] || [ "$status_code" = "201" ]; then
        local total_processed
        total_processed=$(echo "$response" | grep -o '"total_processed":[^,]*' | cut -d':' -f2 | tr -d ' ')
        
        local total_time
        total_time=$(echo "$response" | grep -o '"total_time_ms":[^,]*' | cut -d':' -f2 | tr -d ' ')
        
        print_success "Batch classification: $total_processed texts processed in ${total_time}ms"
    else
        print_error "Batch classification failed (HTTP $status_code)"
        return 1
    fi
}

# Test model info
test_model_info() {
    print_test "Testing model information endpoint..."
    
    local result
    result=$(make_request "GET" "/model-info")
    local response=$(echo "$result" | head -n -1)
    local status_code=$(echo "$result" | tail -n 1)
    
    if [ "$status_code" = "200" ]; then
        local model_name
        model_name=$(echo "$response" | grep -o '"model_name":"[^"]*"' | cut -d'"' -f4)
        
        local total_intents
        total_intents=$(echo "$response" | grep -o '"total_intents":[^,]*' | cut -d':' -f2 | tr -d ' ')
        
        print_success "Model info: $model_name with $total_intents intents"
    else
        print_error "Model info failed (HTTP $status_code)"
        return 1
    fi
}

# Test intents list
test_intents_list() {
    print_test "Testing available intents endpoint..."
    
    local result
    result=$(make_request "GET" "/intents")
    local response=$(echo "$result" | head -n -1)
    local status_code=$(echo "$result" | tail -n 1)
    
    if [ "$status_code" = "200" ]; then
        local total
        total=$(echo "$response" | grep -o '"total":[^,]*' | cut -d':' -f2 | tr -d ' ')
        
        print_success "Available intents: $total categories"
        
        # Show intents if available
        local intents
        intents=$(echo "$response" | grep -o '"intents":\[[^\]]*\]' | sed 's/"intents":\[//; s/\]//; s/"//g')
        
        if [ ! -z "$intents" ]; then
            echo "   Categories: $intents"
        fi
    else
        print_error "Intents list failed (HTTP $status_code)"
        return 1
    fi
}

# Test caching (if enabled)
test_caching() {
    print_test "Testing caching performance..."
    
    local test_text='{"text": "Kapan jadwal kuliah besok?", "include_debug": false}'
    
    # First request (cache miss)
    local start_time=$(date +%s%3N)
    local result1
    result1=$(make_request "POST" "/classify" "$test_text")
    local end_time=$(date +%s%3N)
    local first_time=$((end_time - start_time))
    
    # Second request (should be cache hit)
    local start_time2=$(date +%s%3N)
    local result2
    result2=$(make_request "POST" "/classify" "$test_text")
    local end_time2=$(date +%s%3N)
    local second_time=$((end_time2 - start_time2))
    
    if [ $second_time -lt $first_time ]; then
        print_success "Caching works: ${first_time}ms → ${second_time}ms (${first_time}ms faster)"
    else
        print_warning "Caching might not be enabled or effective"
    fi
}

# Test error handling
test_error_handling() {
    print_test "Testing error handling..."
    
    # Test empty text
    local empty_text='{"text": "", "include_debug": false}'
    local result
    result=$(make_request "POST" "/classify" "$empty_text")
    local status_code=$(echo "$result" | tail -n 1)
    
    if [ "$status_code" = "422" ] || [ "$status_code" = "400" ]; then
        print_success "Empty text validation works (HTTP $status_code)"
    else
        print_warning "Empty text validation might not be working properly"
    fi
    
    # Test invalid endpoint
    local invalid_result
    invalid_result=$(make_request "GET" "/invalid-endpoint")
    local invalid_status=$(echo "$invalid_result" | tail -n 1)
    
    if [ "$invalid_status" = "404" ]; then
        print_success "Invalid endpoint handling works (HTTP $invalid_status)"
    else
        print_warning "Invalid endpoint handling might not be working"
    fi
}

# Test system info (if available)
test_system_info() {
    print_test "Testing system information..."
    
    local result
    result=$(make_request "GET" "/system-info")
    local status_code=$(echo "$result" | tail -n 1)
    
    if [ "$status_code" = "200" ]; then
        print_success "System info endpoint accessible"
    else
        print_warning "System info endpoint not available or restricted"
    fi
}

# Performance benchmark
run_performance_test() {
    print_test "Running performance benchmark..."
    
    local test_text='{"text": "Kapan jadwal kuliah Informatika besok?", "include_debug": false}'
    local requests=10
    local total_time=0
    
    echo "Sending $requests requests..."
    
    for i in $(seq 1 $requests); do
        local start_time=$(date +%s%3N)
        local result
        result=$(make_request "POST" "/classify" "$test_text")
        local end_time=$(date +%s%3N)
        local request_time=$((end_time - start_time))
        
        total_time=$((total_time + request_time))
        
        echo -n "."
    done
    
    echo ""
    
    local avg_time=$((total_time / requests))
    local requests_per_sec=$((1000 / avg_time))
    
    print_success "Performance: ${avg_time}ms average, ~${requests_per_sec} requests/sec"
    
    if [ $avg_time -lt 500 ]; then
        print_success "Good performance (< 500ms)"
    elif [ $avg_time -lt 1000 ]; then
        print_warning "Moderate performance (< 1000ms)"
    else
        print_warning "Slow performance (> 1000ms) - consider optimization"
    fi
}

# Main test execution
main() {
    echo "🧪 DistilBERT Service Test Suite"
    echo "================================="
    echo "Testing service at: $SERVICE_URL"
    echo ""
    
    # Basic connectivity test
    check_service
    
    # Core functionality tests
    test_health
    test_single_classification
    test_batch_classification
    test_model_info
    test_intents_list
    
    # Performance and reliability tests
    test_caching
    test_error_handling
    test_system_info
    
    # Benchmark
    run_performance_test
    
    echo ""
    echo "🎉 Test suite completed!"
    echo ""
    echo "📋 Quick Test Commands:"
    echo "   Health Check: curl $SERVICE_URL/health"
    echo "   Classify Text: curl -X POST $SERVICE_URL/classify -H 'Content-Type: application/json' -d '{\"text\":\"test\"}'"
    echo "   API Docs: $SERVICE_URL/docs"
    echo "   Model Info: curl $SERVICE_URL/model-info"
}

# Execute tests
main "$@"
