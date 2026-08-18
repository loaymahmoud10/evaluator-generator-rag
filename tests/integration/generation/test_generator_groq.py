from app.generation.generator import Generator
from app.generation.llm import create_groq_llm


def test_generator_produces_grounded_answer_with_groq():
    llm = create_groq_llm()

    generator = Generator(llm=llm)

    result = generator.generate(
        question="What is machine learning?",
        retrieved_context=(
            "Machine learning is a branch of artificial intelligence "
            "that allows systems to learn patterns from data and make "
            "predictions or decisions without being explicitly programmed "
            "for every situation."
        ),
    )

    assert isinstance(result, str)
    assert len(result.strip()) > 0

    print("\nGenerated answer:")
    print(result)