#!/usr/bin/env python3
"""
Comprehensive test suite for the Hybrid QA Chatbot System

Tests all components:
- Intent classification
- QA extraction 
- Response generation
- Performance benchmarks
- Error handling
"""

import asyncio
import time
import json
import sys
from typing import List, Dict, Any
import requests
from dataclasses import dataclass

# Test configuration
BASE_URL = "http://localhost:8000"
PERFORMANCE_TARGET_MS = 500

@dataclass
class TestCase:
    message: str
    expected_intent: str
    expected_mode: str
    expected_source: str = None
    description: str = ""

class HybridSystemTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "performance_tests": 0,
            "performance_passed": 0,
            "errors": []
        }
        
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Hybrid QA System Test Suite")
        print("=" * 60)
        
        # Service health check
        if not self.check_service_health():
            print("❌ Service health check failed. Exiting.")
            return False
            
        # Run test categories
        self.test_knowledge_based_intents()
        self.test_conversational_intents() 
        self.test_edge_cases()
        self.test_performance_benchmarks()
        self.test_error_handling()
        
        # Print final results
        self.print_summary()
        
        return self.results["failed"] == 0
        
    def check_service_health(self) -> bool:
        """Check if service is running and healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Service healthy: {health_data.get('status', 'unknown')}")
                return True
            else:
                print(f"❌ Service unhealthy: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Service unreachable: {e}")
            return False
            
    def test_knowledge_based_intents(self):
        """Test knowledge-based intent processing"""
        print("\n📚 Testing Knowledge-Based Intents")
        print("-" * 40)
        
        test_cases = [
            # Product Features
            TestCase(
                message="What are the key features of your chatbot platform?",
                expected_intent="product_features",
                expected_mode="knowledge",
                expected_source="qa_extraction",
                description="Product features inquiry"
            ),
            TestCase(
                message="Tell me about your security features",
                expected_intent="product_security", 
                expected_mode="knowledge",
                expected_source="qa_extraction",
                description="Security features inquiry"
            ),
            
            # Pricing
            TestCase(
                message="How much does the professional plan cost?",
                expected_intent="pricing_professional",
                expected_mode="knowledge", 
                expected_source="qa_extraction",
                description="Professional pricing inquiry"
            ),
            TestCase(
                message="What's included in the enterprise package?",
                expected_intent="pricing_enterprise",
                expected_mode="knowledge",
                expected_source="qa_extraction", 
                description="Enterprise pricing inquiry"
            ),
            
            # Guides
            TestCase(
                message="How do I get started with your platform?",
                expected_intent="setup_guide",
                expected_mode="knowledge",
                expected_source="qa_extraction",
                description="Setup guide inquiry"
            ),
            TestCase(
                message="I need help with integration",
                expected_intent="integration_guide", 
                expected_mode="knowledge",
                expected_source="qa_extraction",
                description="Integration help"
            ),
            
            # Policies
            TestCase(
                message="What is your privacy policy?",
                expected_intent="privacy_policy",
                expected_mode="knowledge",
                expected_source="qa_extraction", 
                description="Privacy policy inquiry"
            ),
            
            # Account
            TestCase(
                message="How do I create a new account?",
                expected_intent="account_creation",
                expected_mode="knowledge",
                expected_source="qa_extraction",
                description="Account creation help"
            ),
            
            # Company
            TestCase(
                message="Tell me about your company",
                expected_intent="company_about",
                expected_mode="knowledge", 
                expected_source="qa_extraction",
                description="Company information inquiry"
            )
        ]
        
        for test_case in test_cases:
            self.run_single_test(test_case)
            
    def test_conversational_intents(self):
        """Test conversational intent processing"""
        print("\n💬 Testing Conversational Intents")
        print("-" * 40)
        
        test_cases = [
            TestCase(
                message="Hello! How are you today?",
                expected_intent="conversational_greeting",
                expected_mode="conversational",
                expected_source="template_response",
                description="Greeting message"
            ),
            TestCase(
                message="Thank you so much for your help!",
                expected_intent="conversational_thanks", 
                expected_mode="conversational",
                expected_source="template_response",
                description="Thank you message"
            ),
            TestCase(
                message="Goodbye, see you later!",
                expected_intent="conversational_goodbye",
                expected_mode="conversational", 
                expected_source="template_response",
                description="Goodbye message"
            ),
            TestCase(
                message="How's the weather today?",
                expected_intent="conversational_chitchat",
                expected_mode="conversational",
                expected_source="template_response",
                description="Casual chitchat"
            ),
            TestCase(
                message="Your service is really great!",
                expected_intent="conversational_feedback",
                expected_mode="conversational",
                expected_source="template_response", 
                description="Positive feedback"
            )
        ]
        
        for test_case in test_cases:
            self.run_single_test(test_case)
            
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print("\n🔍 Testing Edge Cases")
        print("-" * 40)
        
        test_cases = [
            TestCase(
                message="",
                expected_intent="unknown",
                expected_mode="fallback",
                description="Empty message"
            ),
            TestCase(
                message="x" * 1000,  # Very long message
                expected_intent="unknown", 
                expected_mode="fallback",
                description="Extremely long message"
            ),
            TestCase(
                message="123 !@# $%^ &*()",
                expected_intent="unknown",
                expected_mode="fallback", 
                description="Special characters only"
            ),
            TestCase(
                message="price cost money dollar enterprise professional basic",
                expected_intent="pricing_comparison",  # Should match pricing
                expected_mode="knowledge",
                description="Multiple pricing keywords"
            )
        ]
        
        for test_case in test_cases:
            self.run_single_test(test_case)
            
    def test_performance_benchmarks(self):
        """Test response time performance"""
        print("\n⚡ Testing Performance Benchmarks")
        print("-" * 40)
        
        # Test messages with expected fast responses
        test_messages = [
            "What are your pricing plans?",
            "Hello, how are you?", 
            "Tell me about your security features",
            "How do I get started?",
            "Thank you for your help"
        ]
        
        response_times = []
        
        for message in test_messages:
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.base_url}/chat",
                    json={"message": message, "user_id": "test_user"},
                    timeout=10
                )
                
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                self.results["performance_tests"] += 1
                
                if response_time_ms <= PERFORMANCE_TARGET_MS:
                    self.results["performance_passed"] += 1
                    status = "✅"
                else:
                    status = "❌"
                    
                print(f"{status} {message[:30]:<30} | {response_time_ms:.1f}ms")
                
            except Exception as e:
                print(f"❌ Performance test failed: {e}")
                self.results["errors"].append(f"Performance test error: {e}")
                
        # Summary statistics
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            print(f"\n📊 Performance Summary:")
            print(f"   Average: {avg_time:.1f}ms")
            print(f"   Min: {min_time:.1f}ms") 
            print(f"   Max: {max_time:.1f}ms")
            print(f"   Target: <{PERFORMANCE_TARGET_MS}ms")
            print(f"   Passed: {self.results['performance_passed']}/{self.results['performance_tests']}")
            
    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n🛡️ Testing Error Handling")
        print("-" * 40)
        
        # Test invalid request formats
        test_cases = [
            {
                "name": "Missing message field",
                "data": {"user_id": "test"},
                "expect_error": True
            },
            {
                "name": "Invalid JSON",
                "data": "invalid json",
                "expect_error": True  
            },
            {
                "name": "Null message",
                "data": {"message": None, "user_id": "test"},
                "expect_error": False  # Should handle gracefully
            }
        ]
        
        for test_case in test_cases:
            try:
                if isinstance(test_case["data"], str):
                    # Send invalid JSON
                    response = requests.post(
                        f"{self.base_url}/chat",
                        data=test_case["data"],
                        headers={"Content-Type": "application/json"},
                        timeout=5
                    )
                else:
                    response = requests.post(
                        f"{self.base_url}/chat", 
                        json=test_case["data"],
                        timeout=5
                    )
                
                if test_case["expect_error"]:
                    if response.status_code >= 400:
                        print(f"✅ {test_case['name']} - Error handled correctly")
                    else:
                        print(f"❌ {test_case['name']} - Should have returned error")
                        self.results["errors"].append(f"Expected error for: {test_case['name']}")
                else:
                    if response.status_code == 200:
                        print(f"✅ {test_case['name']} - Handled gracefully")
                    else:
                        print(f"❌ {test_case['name']} - Should handle gracefully")
                        
            except Exception as e:
                print(f"⚠️ {test_case['name']} - Exception: {e}")
                
    def run_single_test(self, test_case: TestCase):
        """Run a single test case"""
        self.results["total_tests"] += 1
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "message": test_case.message,
                    "user_id": "test_user",
                    "include_debug": True
                },
                timeout=10
            )
            
            if response.status_code != 200:
                self.results["failed"] += 1
                error_msg = f"HTTP {response.status_code} for: {test_case.description}"
                print(f"❌ {error_msg}")
                self.results["errors"].append(error_msg)
                return
                
            data = response.json()
            
            # Check intent
            intent_match = data.get("intent") == test_case.expected_intent
            
            # Check mode  
            mode_match = data.get("mode") == test_case.expected_mode
            
            # Check source (if specified)
            source_match = True
            if test_case.expected_source:
                source_match = data.get("source") == test_case.expected_source
                
            # Overall test result
            test_passed = intent_match and mode_match and source_match
            
            if test_passed:
                self.results["passed"] += 1
                status = "✅"
            else:
                self.results["failed"] += 1 
                status = "❌"
                
                # Log specific failures
                failures = []
                if not intent_match:
                    failures.append(f"intent: got '{data.get('intent')}', expected '{test_case.expected_intent}'")
                if not mode_match:
                    failures.append(f"mode: got '{data.get('mode')}', expected '{test_case.expected_mode}'")
                if not source_match:
                    failures.append(f"source: got '{data.get('source')}', expected '{test_case.expected_source}'")
                    
                error_msg = f"{test_case.description}: {', '.join(failures)}"
                self.results["errors"].append(error_msg)
            
            # Display result
            confidence = data.get("confidence", 0)
            processing_time = data.get("processing_time_ms", 0)
            
            print(f"{status} {test_case.description:<35} | Intent: {data.get('intent'):<20} | Mode: {data.get('mode'):<12} | {confidence:.2f} conf | {processing_time:.1f}ms")
            
        except Exception as e:
            self.results["failed"] += 1
            error_msg = f"Exception in {test_case.description}: {e}"
            print(f"❌ {error_msg}")
            self.results["errors"].append(error_msg)
            
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total = self.results["total_tests"]
        passed = self.results["passed"] 
        failed = self.results["failed"]
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)" if total > 0 else "Passed: 0")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)" if total > 0 else "Failed: 0")
        
        # Performance summary
        perf_total = self.results["performance_tests"]
        perf_passed = self.results["performance_passed"]
        if perf_total > 0:
            print(f"\nPerformance Tests: {perf_passed}/{perf_total} passed ({perf_passed/perf_total*100:.1f}%)")
            
        # Show errors
        if self.results["errors"]:
            print(f"\n❌ ERRORS ({len(self.results['errors'])}):")
            for i, error in enumerate(self.results["errors"][:10], 1):  # Show first 10 errors
                print(f"   {i}. {error}")
            if len(self.results["errors"]) > 10:
                print(f"   ... and {len(self.results['errors']) - 10} more errors")
                
        # Overall result
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! Hybrid QA system is working correctly.")
        else:
            print(f"\n⚠️ {failed} tests failed. Please review the errors above.")
            
        print("=" * 60)
        
    def test_system_stats(self):
        """Test system statistics endpoint"""
        print("\n📈 Testing System Statistics")
        print("-" * 40)
        
        try:
            response = requests.get(f"{self.base_url}/hybrid-stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print("✅ System statistics available:")
                print(f"   Knowledge intents: {len(stats.get('knowledge_intents', []))}")
                print(f"   Conversational intents: {len(stats.get('conversational_intents', []))}")
                print(f"   QA model loaded: {stats.get('qa_model_loaded', False)}")
                print(f"   Cache enabled: {stats.get('cache_enabled', False)}")
            else:
                print(f"❌ Stats endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Stats test failed: {e}")

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = BASE_URL
        
    tester = HybridSystemTester(base_url)
    
    # Run tests
    success = tester.run_all_tests()
    
    # Test system stats  
    tester.test_system_stats()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
