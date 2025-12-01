from src.rag_chatbot import SimpleRAGChatbot
from src.evaluator import SimpleEvaluator

def test_simple_rag_chatbot():
    text = """
    Hyderabad is the capital of Telangana.
    Bengaluru is known as the Silicon Valley of India.
    """

    bot = SimpleRAGChatbot(text)
    evaluator = SimpleEvaluator()

    query = "Where is Hyderabad?"
    answer = bot.answer(query)
    contexts = bot.retrieve(query)

    # Simple checks
    assert "Hyderabad" in answer
    assert len(contexts) > 0

    # RAGAS score test
    results = evaluator.evaluate(query, answer, contexts)
    for metric, score in results.items():
        assert 0 <= score <= 1
