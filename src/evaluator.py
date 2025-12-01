from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness,
)

class SimpleEvaluator:

    def evaluate(self, question, answer, contexts):
        """Run RAGAS evaluation and return metric scores."""
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }

        return evaluate(
            data,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
        )
