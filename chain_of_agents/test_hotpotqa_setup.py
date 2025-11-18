"""
Test script to verify HotPotQA implementation is working correctly.

This script tests:
1. Data loading functionality
2. Context formatting
3. Answer extraction and normalization
4. BridgeQueryAgents initialization (without running inference)
"""

import sys
sys.path.insert(0, 'chain_of_agents')

from utils import (
    load_hotpotqa_data,
    format_hotpotqa_context,
    extract_hotpotqa_answer,
    normalize_answer,
    check_hotpotqa_exact_match,
    get_bridge_query_prompts
)
from main import BridgeQueryAgents, MajorityVotingAgents


def test_data_loading():
    """Test loading a small subset of HotPotQA data."""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)

    try:
        # Load just 5 examples
        dataset = load_hotpotqa_data(subset_size=5, use_gold_only=False)
        print(f"✓ Successfully loaded {len(dataset)} examples")

        # Check first example structure
        example = dataset[0]
        required_keys = ["question", "answer", "context", "supporting_facts", "type"]
        for key in required_keys:
            assert key in example, f"Missing key: {key}"
        print(f"✓ Example has all required keys: {required_keys}")

        # Display first example
        print(f"\nFirst example:")
        print(f"  Question: {example['question'][:100]}...")
        print(f"  Answer: {example['answer']}")
        print(f"  Context length: {len(example['context'].split())} words")
        print(f"  Type: {example['type']}")

        return True

    except Exception as e:
        print(f"✗ Data loading failed: {str(e)}")
        return False


def test_context_formatting():
    """Test context formatting with both full and gold-only modes."""
    print("\n" + "="*60)
    print("TEST 2: Context Formatting")
    print("="*60)

    try:
        # Create mock context
        mock_context = [
            ["Title 1", ["Sentence 1.", "Sentence 2."]],
            ["Title 2", ["Sentence 3.", "Sentence 4."]],
            ["Title 3", ["Sentence 5.", "Sentence 6."]]
        ]

        mock_supporting_facts = {
            "title": ["Title 1", "Title 3"],
            "sent_id": [0, 1]
        }

        # Test full context
        full_context = format_hotpotqa_context(mock_context, use_gold_only=False)
        print(f"✓ Full context formatted ({len(full_context.split())} words)")

        # Test gold-only context
        gold_context = format_hotpotqa_context(mock_context, mock_supporting_facts, use_gold_only=True)
        print(f"✓ Gold-only context formatted ({len(gold_context.split())} words)")

        # Verify gold context is shorter
        assert len(gold_context) < len(full_context), "Gold context should be shorter"
        print(f"✓ Gold context is shorter than full context")

        return True

    except Exception as e:
        print(f"✗ Context formatting failed: {str(e)}")
        return False


def test_answer_extraction():
    """Test answer extraction and normalization."""
    print("\n" + "="*60)
    print("TEST 3: Answer Extraction & Normalization")
    print("="*60)

    try:
        # Test various response formats
        test_cases = [
            ("Answer: Shirley Temple", "Shirley Temple"),
            ("The answer is: Barack Obama", "Barack Obama"),
            ("The answer is The Beatles", "The Beatles"),
            ("Based on the context, the answer is: 1945", "1945"),
        ]

        for response, expected in test_cases:
            extracted = extract_hotpotqa_answer(response)
            assert expected.lower() in extracted.lower(), f"Failed to extract from: {response}"
            print(f"✓ Extracted '{extracted}' from '{response}'")

        # Test normalization
        test_normalize = [
            ("The Beatles", "beatles"),
            ("  New York City  ", "new york city"),
            ("It's a test!", "its test"),
        ]

        for text, expected in test_normalize:
            normalized = normalize_answer(text)
            assert normalized == expected, f"Normalization failed: {text} -> {normalized} (expected {expected})"
            print(f"✓ Normalized '{text}' to '{normalized}'")

        # Test exact match
        assert check_hotpotqa_exact_match("The Beatles", "Beatles"), "Should match after normalization"
        assert check_hotpotqa_exact_match("New York", "new york"), "Should match case-insensitive"
        assert not check_hotpotqa_exact_match("London", "Paris"), "Should not match different answers"
        print(f"✓ Exact match checking works correctly")

        return True

    except Exception as e:
        print(f"✗ Answer extraction failed: {str(e)}")
        return False


def test_prompt_generation():
    """Test prompt generation for BridgeQueryAgents."""
    print("\n" + "="*60)
    print("TEST 4: Prompt Generation")
    print("="*60)

    try:
        splitter_prompt, updater_prompt, worker_prompt = get_bridge_query_prompts()

        assert "First subquery" in splitter_prompt, "Splitter prompt should mention 'First subquery'"
        print(f"✓ Query splitter prompt generated ({len(splitter_prompt)} chars)")

        assert "Second subquery" in updater_prompt, "Updater prompt should mention 'Second subquery'"
        print(f"✓ Query updater prompt generated ({len(updater_prompt)} chars)")

        assert "Answer:" in worker_prompt, "Worker prompt should mention answer format"
        print(f"✓ Worker prompt generated ({len(worker_prompt)} chars)")

        return True

    except Exception as e:
        print(f"✗ Prompt generation failed: {str(e)}")
        return False


def test_agent_initialization():
    """Test agent initialization (without running inference)."""
    print("\n" + "="*60)
    print("TEST 5: Agent Initialization")
    print("="*60)

    try:
        # Test BridgeQueryAgents initialization
        bridge_agent = BridgeQueryAgents(
            worker_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            manager_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            chunk_size=500,
            max_tokens_worker=512,
            max_tokens_manager=1024
        )
        print(f"✓ BridgeQueryAgents initialized")

        # Test MajorityVotingAgents initialization
        from utils import get_hotpotqa_majority_vote_prompt
        maj_agent = MajorityVotingAgents(
            num_agents=3,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            max_tokens=2048,
            prompt=get_hotpotqa_majority_vote_prompt()
        )
        print(f"✓ MajorityVotingAgents initialized")

        return True

    except Exception as e:
        print(f"✗ Agent initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("HotPotQA Implementation Test Suite")
    print("="*60)

    tests = [
        ("Data Loading", test_data_loading),
        ("Context Formatting", test_context_formatting),
        ("Answer Extraction", test_answer_extraction),
        ("Prompt Generation", test_prompt_generation),
        ("Agent Initialization", test_agent_initialization),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")

    total_passed = sum(1 for _, result in results if result)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    print("="*60 + "\n")

    if total_passed == total_tests:
        print("🎉 All tests passed! Implementation is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
